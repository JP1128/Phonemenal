"""Train a per-patient LoRA adapter on top of Whisper.

The phoneme-bottleneck pipeline can't recover the acoustic detail Whisper
already used. The right place to encode patient-specific distortion
patterns is *inside* Whisper itself — as a small LoRA delta over the base
encoder/decoder. Base weights stay frozen and shared; the only per-patient
artifact is a few MB of adapter weights.

Trains on the deterministic ``cal`` split of ``torgo_manifest`` for one
speaker. Evaluation lives in ``evaluate_adapter.py`` so the training run
stays cheap to re-run while iterating on hyperparameters.

Usage:
    cd server
    .venv/bin/python -m scripts.train_patient_adapter --speaker-id F03
    .venv/bin/python -m scripts.train_patient_adapter --speaker-id M01 --epochs 10
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import hashlib  # noqa: E402

from app.config import settings  # noqa: E402
from app.db import (  # noqa: E402
    connect,
    get_calibration,
    list_calibration_utterances,
    list_corrective_training_pairs,
    list_torgo_clips,
    update_training_job,
)
from scripts.splits import filter_by_split  # noqa: E402


# Inner train/val split lives entirely *inside* the cal split. A different
# hash seed than the outer cal/eval split guarantees no overlap with eval —
# eval is held out at the dataset level; val is held out at training time so
# early stopping can detect overfitting before it touches the evaluation set.
_VAL_SEED = 4242


def _inner_train_val_split(
    clips: list[dict], val_frac: float = 0.1, seed: int = _VAL_SEED
) -> tuple[list[dict], list[dict]]:
    train: list[dict] = []
    val: list[dict] = []
    for c in clips:
        key = f"{seed}|{c['speaker_id']}|{c['id']}".encode("utf-8")
        bucket = int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / (1 << 64)
        (val if bucket < val_frac else train).append(c)
    return train, val

logger = logging.getLogger("train_patient_adapter")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")


TARGET_SR = 16000


def _load_audio(path: str) -> np.ndarray | None:
    """Decode wav → float32 mono at 16 kHz; return None on failure or empty."""
    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception as exc:
        logger.warning("audio load failed for %s: %s", path, exc)
        return None
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != TARGET_SR:
        import librosa  # heavy, only when needed

        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    if audio.size == 0:
        return None
    return audio.astype(np.float32, copy=False)


def _build_dataset(
    clips: list[dict],
    processor,
    return_balance_weights: bool = False,
    phoneme_extractor=None,
):
    """Materialize (input_features, labels) pairs from a list of clip rows.

    Whisper's processor handles mel-spec extraction on the audio side and
    BPE tokenization on the text side. Labels use ``-100`` on padded
    positions so the loss ignores them — the data collator will substitute
    the pad token id with -100 at batch time.

    When ``return_balance_weights=True``, also returns a parallel list of
    per-clip sampling weights set to ``1 / max(1, word_count)``. With per-token
    cross-entropy, this equalizes the expected per-token contribution of each
    clip — the fix for the prompt-set imbalance where TIMIT clips (~7.5 words
    each) were drowning out isolated-word clips (1 word each) in training.

    When ``phoneme_extractor`` is provided, also pre-computes wav2vec2
    phoneme posteriors for each clip and stores them as ``phoneme_targets``
    (shape ``[1500, V_phon]``). The audio is padded to 30s before extraction
    so the time axis matches Whisper's encoder output frame-for-frame.
    """
    from datasets import Dataset

    rows = []
    weights: list[float] = []
    target_audio_len = 16000 * 30  # Whisper's 30s window
    for clip in clips:
        text = (clip.get("intended_text") or "").strip()
        if not text:
            continue
        audio = _load_audio(clip["wav_path"])
        if audio is None:
            continue
        row = {"audio": audio, "text": text}
        if phoneme_extractor is not None:
            if audio.size < target_audio_len:
                padded = np.pad(audio, (0, target_audio_len - audio.size))
            else:
                padded = audio[:target_audio_len]
            # float16 cuts the cached target memory in half (1500*392*4=2.35MB
            # per clip → 1.17MB) without measurable accuracy impact in soft-
            # target distillation.
            probs = phoneme_extractor.extract(padded, TARGET_SR)
            row["phoneme_targets"] = probs.astype(np.float16)
        rows.append(row)
        n_words = max(1, len(text.split()))
        weights.append(1.0 / n_words)

    if not rows:
        raise SystemExit("no usable training rows after audio decode")

    def _prepare(batch):
        feats = processor.feature_extractor(
            batch["audio"], sampling_rate=TARGET_SR, return_tensors="np"
        )
        batch["input_features"] = feats.input_features[0]
        labels = processor.tokenizer(batch["text"]).input_ids
        batch["labels"] = labels
        return batch

    ds = Dataset.from_list(rows)
    keep_cols = ["phoneme_targets"] if phoneme_extractor is not None else []
    ds = ds.map(_prepare, remove_columns=["audio", "text"])
    if return_balance_weights:
        return ds, weights
    return ds


class _PhonemeAuxModule(torch.nn.Module):
    """Wraps a PEFT-Whisper model + a small phoneme head.

    Both submodules are registered under one ``nn.Module`` so the optimizer
    collects gradients from both. The combined loss is computed *inside*
    forward — when phoneme_targets is provided in the batch, the encoder
    hidden states are fed through the aux head, KL divergence with the
    teacher is computed, and the returned ``outputs.loss`` is the combined
    text-CE + phoneme-KL loss. Doing it in forward (rather than in a custom
    Trainer.compute_loss) means standard HF Trainer evaluation paths pick
    up the combined loss automatically.

    The aux head is purely a training-time regularizer — it doesn't change
    inference. After training, save_pretrained() saves only the LoRA weights;
    the aux head is discarded.
    """

    def __init__(
        self,
        peft_whisper,
        phoneme_vocab_size: int,
        encoder_hidden_size: int,
        aux_weight: float,
    ):
        super().__init__()
        self.whisper = peft_whisper
        self.aux_head = torch.nn.Linear(encoder_hidden_size, phoneme_vocab_size)
        self.aux_weight = float(aux_weight)
        # Whisper config delegation so HuggingFace tooling treats this like
        # the underlying model.
        self.config = peft_whisper.config

    def forward(self, *args, phoneme_targets=None, **kwargs):
        # Force encoder hidden states for the aux loss path. Cheap to keep on
        # always — the encoder runs once anyway.
        if phoneme_targets is not None:
            kwargs["output_hidden_states"] = True
            kwargs["return_dict"] = True

        # transformers 5.8 + peft 0.19 regression: calling the PeftModel
        # wrapper's __call__ with ``labels=`` raises
        #   TypeError: WhisperDecoder(...) got multiple values for
        #   keyword argument 'input_ids'
        # because PeftModel.__call__ pre-derives ``decoder_input_ids``
        # from labels and forwards them, but the Whisper forward in
        # transformers 5.8 also derives them internally — the two paths
        # collide.
        #
        # Workaround: call the inner WhisperForConditionalGeneration
        # directly. PEFT substitutes ``lora.Linear`` modules *inside*
        # that base model, so LoRA still applies on the forward — we
        # only lose the PeftModel-level bookkeeping (adapter switching,
        # disable_adapter() context, etc.) which we don't use here.
        inner = getattr(self.whisper, "base_model", None)
        inner = getattr(inner, "model", None) if inner is not None else None
        outputs = inner(*args, **kwargs) if inner is not None else self.whisper(*args, **kwargs)
        if phoneme_targets is None or self.aux_weight <= 0:
            return outputs

        import torch.nn.functional as F

        encoder_last = outputs.encoder_hidden_states[-1]
        aux_logits = self.aux_head(encoder_last)
        T = min(aux_logits.size(1), phoneme_targets.size(1))
        log_probs = F.log_softmax(aux_logits[:, :T, :], dim=-1)
        tgt = phoneme_targets[:, :T, :]
        # mean over batch and time so the per-frame KL scale matches per-token
        # text CE — without this the 1500-frame summed KL dominates by ~1500x.
        kl_per_frame = F.kl_div(
            log_probs, tgt, reduction="none", log_target=False
        )
        aux_loss = kl_per_frame.sum(dim=-1).mean()
        text_loss = outputs.loss
        outputs.loss = (1 - self.aux_weight) * text_loss + self.aux_weight * aux_loss
        return outputs

    # ---- delegation: gradient checkpointing, save, trainable-param print ----
    def gradient_checkpointing_enable(self, *args, **kwargs):
        if hasattr(self.whisper, "gradient_checkpointing_enable"):
            self.whisper.gradient_checkpointing_enable(*args, **kwargs)

    def enable_input_require_grads(self):
        if hasattr(self.whisper, "enable_input_require_grads"):
            self.whisper.enable_input_require_grads()

    def save_pretrained(self, path, **kwargs):
        # Aux head is throwaway — keep checkpoints adapter-only.
        self.whisper.save_pretrained(path, **kwargs)

    def print_trainable_parameters(self):
        if hasattr(self.whisper, "print_trainable_parameters"):
            self.whisper.print_trainable_parameters()
        n_aux = sum(p.numel() for p in self.aux_head.parameters() if p.requires_grad)
        logger.info("+ phoneme aux_head: %s trainable params", f"{n_aux:,}")


class _Collator:
    """Pad input_features and labels to the longest in the batch.

    Whisper's mel features are fixed-length (3000 frames) after the feature
    extractor, so input_features only needs stacking. Labels are variable
    length token sequences and need right-padding with -100 (the loss-ignore
    sentinel) — pad token id replacement happens here, not at tokenize time.
    """

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_feats = torch.tensor(
            np.stack([f["input_features"] for f in features]), dtype=torch.float32
        )
        label_lists = [f["labels"] for f in features]
        max_len = max(len(x) for x in label_lists)
        pad_id = self.processor.tokenizer.pad_token_id
        labels = np.full((len(label_lists), max_len), pad_id, dtype=np.int64)
        for i, ids in enumerate(label_lists):
            labels[i, : len(ids)] = ids
        labels_t = torch.tensor(labels)
        labels_t = labels_t.masked_fill(labels_t == pad_id, -100)
        # Whisper expects the BOS prefix to be stripped from labels; its
        # `forced_decoder_ids` handle decoder priming. If the tokenizer
        # already prepended <|startoftranscript|>, drop it so loss isn't
        # computed on the language tag.
        if (labels_t[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels_t = labels_t[:, 1:]
        out = {"input_features": input_feats, "labels": labels_t}
        if "phoneme_targets" in features[0]:
            stacked = np.stack([f["phoneme_targets"] for f in features])
            out["phoneme_targets"] = torch.tensor(stacked, dtype=torch.float32)
        return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a per-speaker LoRA adapter on Whisper using TORGO cal-split clips."
    )
    parser.add_argument(
        "--speaker-id",
        default=None,
        help="Single-speaker mode: e.g. F03, M01, FC02. Mutually exclusive with "
        "--include-condition.",
    )
    parser.add_argument(
        "--calibration-id",
        type=int,
        default=None,
        help="Train on a clinical calibration (data collected via the "
        "/calibrate page) instead of TORGO. Mutually exclusive with "
        "--speaker-id / --include-condition. Output adapter goes to "
        "data/adapters/cal_<id>/ by default — see --output-name to "
        "override the slug.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help="Override the adapter slug used to build the default "
        "--output-dir. Useful with --calibration-id (e.g. "
        "--output-name jane_doe → data/adapters/jane_doe/).",
    )
    parser.add_argument(
        "--include-feedback-slug",
        default=None,
        help="When set together with --calibration-id, pull confirmed/"
        "corrected live utterances tagged with this adapter slug back "
        "into the training set. These rows are stamped on utterances by "
        "the WS handler at ingest time, so the trainer can include "
        "(audio, listener-validated text) pairs alongside the prompted "
        "calibration clips. Empty / unmatched slug = no-op.",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        default=None,
        help="Internal: training_jobs row to update on start/complete/fail. "
        "Set by the FastAPI training endpoint when spawning this script as a "
        "subprocess so the parent process can poll progress without an IPC "
        "channel.",
    )
    parser.add_argument(
        "--include-condition",
        choices=("control", "dysarthria"),
        default=None,
        help="Pool mode: train on every speaker matching this condition. Used "
        "for stage-1 of two-stage personalization (a generic-dysarthria adapter).",
    )
    parser.add_argument(
        "--exclude-speakers",
        default="",
        help="Comma-separated speaker_ids to exclude from --include-condition. "
        "Used for leave-one-out stage-1 training so the held-out patient is "
        "evaluated on a stage-1 model that never saw their data.",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        help="Path to an existing LoRA adapter directory to warm-start from. "
        "When set, the LoRA weights are loaded as the initial adapter state "
        "instead of random init — used for stage-2 of two-stage training.",
    )
    parser.add_argument(
        "--base-model",
        default="openai/whisper-small.en",
        help="HuggingFace Whisper checkpoint to adapt. Default: whisper-small.en (244M).",
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank. 16 is a common starting point; lower = smaller, less expressive.",
    )
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--extended-modules",
        action="store_true",
        help="Target q/k/v/o + fc1/fc2 instead of just q/v. ~5x more trainable "
        "params (adapter goes from ~1MB to ~5MB). More capacity for severe cases.",
    )
    parser.add_argument(
        "--use-dora",
        action="store_true",
        help="Use DoRA (Decomposed LoRA, 2024) instead of vanilla LoRA. "
        "Same parameter count; decomposes the weight update into magnitude + "
        "direction. Reliably +2-5%% relative on speech in published results.",
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Sample training clips with weight 1/word_count so per-token loss "
        "contribution is equalized across prompt sets. Without this, TIMIT "
        "(~7.5 words/clip) dominates over isolated-word stimuli (1 word/clip), "
        "biasing the LoRA toward 'produce sentences' priors and breaking "
        "isolated-word recognition at decode time.",
    )
    parser.add_argument(
        "--phoneme-aux-weight",
        type=float,
        default=0.0,
        help="When > 0, add an auxiliary phoneme-prediction loss. The aux "
        "objective is: a small Linear head on Whisper's encoder output "
        "predicts wav2vec2's frame-level phoneme posteriors via KL "
        "divergence (soft-target distillation). Combined loss is "
        "(1-w)*text_CE + w*phoneme_KL. Typical w=0.1-0.3. Acoustic-content "
        "regularizer that doesn't suffer from prompt-set token imbalance.",
    )
    parser.add_argument(
        "--phoneme-model",
        default=None,
        help="Wav2vec2 phoneme model for the auxiliary teacher signal. "
        "Default: settings.phoneme_model_name (the existing project model).",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Trade compute for memory by re-running forward during backward. "
        "Required for whisper-small/medium on MPS at non-trivial batch sizes — "
        "without it, peak memory blows past available RAM and the process "
        "swap-thrashes (we hit 26 GB peak on whisper-small batch=8). Pair "
        "with smaller --batch-size and larger --grad-accum.",
    )
    parser.add_argument(
        "--max-cal-clips",
        type=int,
        default=0,
        help="Cap training clips for fast iteration. 0 = all cal-split clips.",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of cal-split clips held out for early stopping.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help="Stop training when val loss has not improved for this many "
        "evaluations. Set to 0 to disable early stopping (fixed-epoch run).",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=20,
        help="Evaluate val loss every N optimizer steps.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Adapter output dir. Default: data/adapters/<speaker_id>/.",
    )
    parser.add_argument("--db", default=settings.db_path)
    parser.add_argument(
        "--device",
        default=None,
        help="Override device: cpu/mps/cuda. Default: auto (mps if avail, else cpu).",
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    logger.info("device=%s base=%s", device, args.base_model)

    sources_set = sum(
        x is not None
        for x in (args.speaker_id, args.include_condition, args.calibration_id)
    )
    if sources_set != 1:
        raise SystemExit(
            "exactly one of --speaker-id / --include-condition / "
            "--calibration-id is required"
        )

    excluded = {s.strip() for s in args.exclude_speakers.split(",") if s.strip()}

    if args.speaker_id is not None:
        run_label = args.speaker_id
        default_dir = f"data/adapters/{args.speaker_id}"
    elif args.calibration_id is not None:
        slug = (args.output_name or f"cal_{args.calibration_id}").strip()
        # Filesystem-safe slug — no path separators, no whitespace in dirs.
        slug = (
            "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in slug)
            .strip("_")
            or f"cal_{args.calibration_id}"
        )
        run_label = slug
        default_dir = f"data/adapters/{slug}"
    else:
        suffix = (
            "_no_" + "_".join(sorted(excluded))
            if excluded
            else "_all"
        )
        run_label = f"pool_{args.include_condition}{suffix}"
        default_dir = f"data/adapters/{run_label}"

    output_dir = Path(args.output_dir or default_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    conn = connect(args.db)

    # Mark the row as running so the parent polling endpoint flips off
    # "queued". The subprocess owns the DB row from this point until it
    # writes a terminal status (succeeded / failed) at the end of main().
    if args.job_id is not None:
        import os
        try:
            update_training_job(conn, args.job_id, status="running", pid=os.getpid())
        except Exception:  # noqa: BLE001 — never let DB issues kill training
            logger.warning("could not update training_jobs row %s", args.job_id)

    # Pull cal-split clips. Single-speaker mode filters by speaker_id; pool mode
    # filters by condition and post-filters out the leave-one-out speakers;
    # calibration-id mode reads from calibration_utterances instead and
    # reshapes those rows to match the TORGO clip dict shape.
    if args.speaker_id is not None:
        all_clips = list_torgo_clips(
            conn, speaker_id=args.speaker_id, limit=1_000_000
        )
    elif args.calibration_id is not None:
        cal_row = get_calibration(conn, args.calibration_id)
        if cal_row is None:
            raise SystemExit(f"calibration {args.calibration_id} not found")
        cal_speaker = (
            (cal_row.get("display_name") or "").strip()
            or f"cal_{args.calibration_id}"
        )
        utterances = list_calibration_utterances(conn, args.calibration_id)
        all_clips = []
        for u in utterances:
            if not u.get("audio_path"):
                continue
            if not (u.get("intended_text") or "").strip():
                continue
            all_clips.append(
                {
                    "id": int(u["id"]),
                    "speaker_id": cal_speaker,
                    "intended_text": u["intended_text"],
                    "prompt_set": u.get("sentence_set_id") or "unknown",
                    "audio_path": u["audio_path"],
                    "duration_ms": u.get("duration_ms") or 0,
                }
            )
        logger.info(
            "calibration %s (label=%r) → %d usable clips",
            args.calibration_id, cal_speaker, len(all_clips),
        )

        # Pull live utterances that were confirmed / corrected by a
        # listener while *this* patient's adapter was active. Each one
        # is a free (audio, listener-validated text) pair, so it lands
        # in the training set alongside the prompted clips. The unique
        # id namespace is shared with calibration utterance ids — same
        # autoincrement table is never used for both — so we prefix to
        # avoid collisions in `_inner_train_val_split` / split hashing.
        if args.include_feedback_slug:
            feedback_rows = list_corrective_training_pairs(
                conn, args.include_feedback_slug
            )
            kept = 0
            for r in feedback_rows:
                ap = r.get("audio_path") or ""
                if not ap or not Path(ap).exists():
                    continue
                all_clips.append(
                    {
                        # Keep ids well-separated from calibration_utterances
                        # ids so the split hashes are distinct.
                        "id": 10_000_000 + int(r["id"]),
                        "speaker_id": cal_speaker,
                        "intended_text": r["intended_text"],
                        # Tag feedback clips with their own prompt_set so
                        # the balanced sampler treats them as a separate
                        # bucket and so they're easy to spot in logs.
                        "prompt_set": f"feedback_{r['source']}",
                        "audio_path": ap,
                        "duration_ms": r.get("duration_ms") or 0,
                    }
                )
                kept += 1
            if kept:
                logger.info(
                    "feedback slug=%r → +%d live clips merged into training set",
                    args.include_feedback_slug, kept,
                )
    else:
        all_clips = list_torgo_clips(
            conn, condition=args.include_condition, limit=1_000_000
        )
        if excluded:
            before = len(all_clips)
            all_clips = [c for c in all_clips if c["speaker_id"] not in excluded]
            logger.info(
                "excluding %d clips from speakers %s (had %d, kept %d)",
                before - len(all_clips), sorted(excluded), before, len(all_clips),
            )

    cal = [
        c for c in filter_by_split(all_clips, "cal")
        if (c.get("intended_text") or "").strip()
        and c.get("prompt_set") != "non-word"
    ]
    if args.max_cal_clips:
        cal = cal[: args.max_cal_clips]
    if not cal:
        raise SystemExit(
            f"no cal-split clips with intended_text for speaker {args.speaker_id!r}"
        )

    train_clips, val_clips = _inner_train_val_split(cal, val_frac=args.val_frac)
    if not val_clips:
        raise SystemExit(
            f"no validation clips after inner split (val_frac={args.val_frac}); "
            "too few cal clips to early-stop. Lower --val-frac or set "
            "--early-stopping-patience 0."
        )
    total_min = sum(c.get("duration_ms") or 0 for c in train_clips) / 60_000
    val_min = sum(c.get("duration_ms") or 0 for c in val_clips) / 60_000
    logger.info(
        "run=%s train=%d clips (~%.1f min)  val=%d clips (~%.1f min)",
        run_label,
        len(train_clips),
        total_min,
        len(val_clips),
        val_min,
    )

    from transformers import (  # heavy imports deferred
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        WhisperForConditionalGeneration,
        WhisperProcessor,
    )
    from peft import LoraConfig, PeftModel, get_peft_model

    processor = WhisperProcessor.from_pretrained(args.base_model)
    base_model = WhisperForConditionalGeneration.from_pretrained(args.base_model)
    # Whisper's generation config sets forced_decoder_ids that conflict with
    # supervised training; clear them so labels drive the loss, not priming.
    base_model.config.forced_decoder_ids = None
    base_model.config.suppress_tokens = []
    # Disable cache during training (incompatible with gradient checkpointing).
    base_model.config.use_cache = False

    if args.init_from:
        # Warm-start from a previously trained adapter (stage-2 of two-stage
        # training). PEFT loads the rank, alpha, target_modules, dropout
        # from the saved adapter_config.json — so any --lora-* flags are
        # ignored when --init-from is set, by design.
        init_path = Path(args.init_from).resolve()
        if not init_path.exists():
            raise SystemExit(f"--init-from path does not exist: {init_path}")
        logger.info("warm-starting LoRA weights from %s", init_path)
        model = PeftModel.from_pretrained(
            base_model, str(init_path), is_trainable=True
        )
    else:
        # NOTE: task_type is intentionally omitted. Setting SEQ_2_SEQ_LM makes
        # PEFT auto-prepare ``input_ids`` for the wrapped model, but Whisper's
        # forward takes ``input_features`` (mel spec) and explodes with
        # "got multiple values for keyword argument 'input_ids'".
        target_modules = (
            ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
            if args.extended_modules
            else ["q_proj", "v_proj"]
        )
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_dora=args.use_dora,
        )
        model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    if args.gradient_checkpointing:
        # PEFT-wrapped models need explicit input-grad requirement; otherwise
        # the input mel-spec tensor has requires_grad=False and gradient
        # checkpointing silently breaks the backward graph.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    aux_extractor = None
    if args.phoneme_aux_weight > 0:
        from app.config import settings as _settings
        from app.phonemes import PhonemeExtractor

        ph_model = args.phoneme_model or _settings.phoneme_model_name
        logger.info(
            "loading phoneme teacher (%s) for auxiliary loss (w=%.2f)…",
            ph_model, args.phoneme_aux_weight,
        )
        aux_extractor = PhonemeExtractor(ph_model, device=device)

    if args.balanced_sampling:
        train_ds, train_weights = _build_dataset(
            train_clips, processor,
            return_balance_weights=True,
            phoneme_extractor=aux_extractor,
        )
    else:
        train_ds = _build_dataset(
            train_clips, processor, phoneme_extractor=aux_extractor
        )
        train_weights = None
    val_ds = _build_dataset(val_clips, processor, phoneme_extractor=aux_extractor)

    if aux_extractor is not None:
        # Wrap in PhonemeAuxModule so the aux head is a registered submodule
        # AND so the combined loss is computed inside forward (which standard
        # Trainer eval paths pick up automatically). Done after dataset build
        # so the wav2vec2 teacher can be reused for phoneme-target extraction.
        ph_vocab = aux_extractor.model.config.vocab_size
        encoder_hidden = base_model.config.d_model
        model = _PhonemeAuxModule(
            model, ph_vocab, encoder_hidden, aux_weight=args.phoneme_aux_weight
        )
        model.print_trainable_parameters()

    # Estimate total steps so warmup is sized to the actual run.
    steps_per_epoch = max(
        1, len(train_ds) // (args.batch_size * max(args.grad_accum, 1))
    )
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, total_steps // 10)
    early_stop = args.early_stopping_patience > 0
    # Auxiliary-loss path interacts badly with HF Trainer's eval loop —
    # the wrapped module's outputs.loss isn't surfacing as `eval_loss` in
    # the metrics dict, which crashes EarlyStoppingCallback. Disable
    # early stopping in that case and just train fixed epochs.
    if early_stop and aux_extractor is not None:
        logger.info(
            "disabling early stopping for aux-loss run "
            "(HF Trainer eval path doesn't surface combined loss)"
        )
        early_stop = False

    # Wrapper module + Trainer's load_best_model_at_end don't compose cleanly
    # (the reload path assumes PreTrainedModel.from_pretrained semantics that
    # our wrapper doesn't provide). With aux loss enabled, just keep the
    # final-step model — early stopping still controls when to stop, but we
    # don't restore the best checkpoint.
    load_best = early_stop and aux_extractor is None

    train_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_steps=warmup_steps,
        logging_steps=10,
        eval_strategy="steps" if early_stop else "no",
        eval_steps=args.eval_steps,
        save_strategy="steps" if load_best else "no",
        save_steps=args.eval_steps,
        save_total_limit=2 if load_best else None,
        load_best_model_at_end=load_best,
        # EarlyStoppingCallback requires metric_for_best_model even when we
        # don't restore the best model — it uses it to track which checkpoint
        # is "best" for patience-decision purposes.
        metric_for_best_model="eval_loss" if early_stop else None,
        greater_is_better=False,
        gradient_checkpointing=args.gradient_checkpointing,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,  # MPS dataloader workers are flaky
        # MPS doesn't reliably support fp16/bf16 across all ops; fp32 is safer.
        fp16=False,
        bf16=False,
        # Use the auto device pick above.
        use_cpu=(device == "cpu"),
    )

    callbacks = []
    if early_stop:
        from transformers import EarlyStoppingCallback

        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        )

    # Subclass Seq2SeqTrainer to work around the transformers-5.8 +
    # peft-0.19 regression where ``PeftModel.__call__(input_features=…,
    # labels=…)`` raises
    #   TypeError: WhisperDecoder(...) got multiple values for keyword
    #   argument 'input_ids'
    # because PeftModel pre-derives ``decoder_input_ids`` and the
    # Whisper forward also derives them internally — the two collide.
    # Bypass: call the inner WhisperForConditionalGeneration directly.
    # PEFT substitutes ``lora.Linear`` modules *inside* that base model
    # at load time, so LoRA still applies on the forward — we only
    # lose PeftModel-level adapter bookkeeping (which we don't use
    # during training, since there is one active adapter per run).
    # For ``_PhonemeAuxModule``-wrapped models, this same bypass lives
    # inside ``_PhonemeAuxModule.forward``, so the override below
    # passes them through untouched.
    class _BypassPeftLossTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, return_outputs=False,
                          num_items_in_batch=None):
            inner = getattr(model, "base_model", None)
            inner = getattr(inner, "model", None) if inner is not None else None
            target = inner if inner is not None else model
            outputs = target(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return (loss, outputs) if return_outputs else loss

    base_trainer_cls = _BypassPeftLossTrainer
    # Aux loss is computed inside _PhonemeAuxModule.forward; no Trainer
    # subclass needed for that. Standard Trainer paths see the combined
    # outputs.loss and treat it like any other loss.

    if train_weights is not None:
        # Subclass to inject a WeightedRandomSampler. Trainer normally builds a
        # RandomSampler from the dataset length; we need a sampler that uses
        # our per-clip weights so per-token loss contribution is balanced.
        from torch.utils.data import WeightedRandomSampler

        weights_tensor = torch.tensor(train_weights, dtype=torch.float64)

        class BalancedTrainer(base_trainer_cls):
            def _get_train_sampler(self, *args, **kwargs):  # signature stable
                return WeightedRandomSampler(
                    weights=weights_tensor,
                    num_samples=len(train_ds),
                    replacement=True,
                )

        trainer_cls = BalancedTrainer
        logger.info(
            "balanced sampling enabled: weight = 1/word_count per clip"
        )
    else:
        trainer_cls = base_trainer_cls

    trainer = trainer_cls(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if early_stop else None,
        data_collator=_Collator(processor),
        processing_class=processor.feature_extractor,
        callbacks=callbacks,
    )

    logger.info(
        "starting training: %d epochs, batch_size=%d, accum=%d, lr=%.0e, early_stop=%s",
        args.epochs, args.batch_size, args.grad_accum, args.lr,
        f"patience={args.early_stopping_patience}" if early_stop else "off",
    )
    trainer.train()

    # Save only the LoRA delta; base weights are recoverable from --base-model.
    model.save_pretrained(str(output_dir))
    processor.save_pretrained(str(output_dir))
    (output_dir / "BASE_MODEL.txt").write_text(args.base_model + "\n")
    logger.info("adapter saved to %s", output_dir)

    # Per-patient threshold calibration. Runs the freshly-saved adapter
    # through the gated-transcribe pipeline on cal-split clips, captures
    # the patient's natural distribution of gate signals (align score,
    # voiced-duration density, per-word & overall confidence), and
    # writes percentile-derived thresholds to
    # ``{output_dir}/calibration.json``. The live runtime's
    # GatedTranscriber + _process_patient consume this so the alignment
    # / density / low-confidence thresholds are patient-specific
    # instead of global magic numbers. Failures are non-fatal — the
    # adapter is still usable with global defaults.
    try:
        from scripts.compute_calibration import compute_calibration
        logger.info("computing per-patient threshold calibration…")
        compute_calibration(
            speaker_id=args.speaker_id or "",
            adapter_dir=output_dir,
            base_model=args.base_model,
            max_clips=50,
        )
    except Exception:  # noqa: BLE001
        logger.exception("calibration step failed (non-fatal)")

    # Final job-status update. Done last so any earlier failure (data,
    # decode, OOM, save) propagates up to the except handler below and
    # is recorded as 'failed' with an error message.
    if args.job_id is not None:
        import time as _t

        try:
            update_training_job(
                conn,
                args.job_id,
                status="succeeded",
                completed_at=_t.time(),
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "could not mark training job %s as succeeded", args.job_id
            )
    return 0


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except SystemExit:
        # Argparse / explicit raises; let it propagate after status update.
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("training run failed")
        # Best-effort: find the job id by re-parsing argv (the main()
        # connection scope may not be reachable here).
        try:
            import os, time as _t
            argv = sys.argv
            if "--job-id" in argv:
                jid = int(argv[argv.index("--job-id") + 1])
                from app.db import connect as _connect, update_training_job as _upd
                _c = _connect(settings.db_path)
                _upd(
                    _c,
                    jid,
                    status="failed",
                    error_message=f"{type(e).__name__}: {e}",
                    completed_at=_t.time(),
                )
        except Exception:  # noqa: BLE001
            pass
        sys.exit(1)
    sys.exit(rc)
