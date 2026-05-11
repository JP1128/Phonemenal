"""Wav2Vec2 phoneme probability extraction (Layer 3 of the pipeline)."""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCTC, AutoProcessor


# wav2vec2's conv stack downsamples by 320 → 20 ms output stride at 16 kHz.
FRAME_STRIDE_SAMPLES = 320


class PhonemeExtractor:
    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForCTC.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self._id_to_token: dict[int, str] = {
            i: tok for tok, i in self.processor.tokenizer.get_vocab().items()
        }

    @torch.inference_mode()
    def extract(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Return softmax probabilities, shape ``[T, V]``.

        ``T`` is the number of 20 ms frames; ``V`` is the phoneme vocabulary size
        (espeak phoneme inventory plus CTC special tokens).
        """
        inputs = self.processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs.input_values.to(self.device)
        logits = self.model(input_values).logits
        return torch.softmax(logits, dim=-1)[0].cpu().numpy()

    def top_k_per_frame(
        self, probs: np.ndarray, k: int
    ) -> list[list[tuple[str, float]]]:
        """Return the top-k ``(token, prob)`` pairs per frame, sorted desc."""
        k = min(k, probs.shape[-1])
        out: list[list[tuple[str, float]]] = []
        for frame in probs:
            idxs = np.argpartition(frame, -k)[-k:]
            idxs = idxs[np.argsort(-frame[idxs])]
            out.append([(self._id_to_token[int(i)], float(frame[i])) for i in idxs])
        return out
