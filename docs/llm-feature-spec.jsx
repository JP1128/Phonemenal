import { useState } from "react";

// ─── Complete LLM Feature Specification ───

const INPUT_FEATURES = [
  {
    id: "phoneme-probs",
    title: "Phoneme Probability Sequences",
    priority: "Critical",
    icon: "🎯",
    color: "#ef4444",
    description: "The raw output from the acoustic layer — not 'what was said' but 'what sounds were attempted'",
    details: [
      "Each time-slice produces a probability distribution over ~44 English phonemes",
      "The LLM receives the full distribution, not a hard label — e.g., {/θ/: 0.30, /d/: 0.42, /t/: 0.18, ∅: 0.10}",
      "Sequences include timing information: duration of each phoneme attempt, gaps between them",
      "Partial phoneme attempts are preserved — a half-formed /k/ that trails off is distinct from silence",
      "Co-articulation context: how each phoneme was influenced by adjacent sounds",
    ],
    whyLLM: "The LLM needs the full probability space to perform soft decoding. If the acoustic layer says 'this is 42% /d/ and 30% /θ/', and the patient's distortion model says they consistently map /θ/ → /d/, the LLM can recover the intended /θ/ with high confidence. Hard labels destroy this signal permanently.",
    example: "Patient says 'I dink I nee some wah-er'\n→ Phoneme layer sends: [aɪ] [d(0.42)/θ(0.30)/t(0.18)] [ɪ] [ŋ] [k] [aɪ] [n] [iː] [s] [ʌ] [m] [w] [ɑː] [∅(pause:340ms)] [ɚ]\n→ LLM resolves: 'I think I need some water'",
  },
  {
    id: "distortion-model",
    title: "Patient Distortion Profile",
    priority: "Critical",
    icon: "🧬",
    color: "#f59e0b",
    description: "The learned model of THIS patient's specific speech impediment patterns, built during calibration",
    details: [
      "Phoneme confusion matrix: P(intended_phoneme | produced_phoneme) — the core mapping",
      "Positional rules: distortions that differ based on word position (initial, medial, final)",
      "Consonant cluster rules: which clusters get simplified and how (e.g., /str/ → /sr/, /pl/ → /p/)",
      "Vowel drift patterns: systematic shifts in vowel production (e.g., /iː/ → /ɪ/ consistently)",
      "Syllable stress errors: which syllable positions tend to get dropped or weakened",
      "Compensatory strategies: alternative productions the patient has developed (e.g., using /f/ when /θ/ fails)",
    ],
    whyLLM: "Without this profile, the LLM is guessing blindly — it would just be a generic speech corrector. With it, the LLM knows that when THIS patient produces /d/, there's a 70% chance they meant /θ/, a 15% chance they meant /d/, and a 15% chance they meant /t/. That specificity is what makes translation accurate.",
    example: "Patient Profile #0042:\n• /θ/ → /d/ (85% word-initial, 60% word-final)\n• /r/ → /w/ (90% all positions)\n• /s/ clusters: /sp/ → /p/, /st/ → /t/, /sk/ → /k/\n• Drops unstressed first syllables 40% of the time\n• Compensates /θ/ with /f/ when self-correcting",
  },
  {
    id: "fatigue-state",
    title: "Fatigue & Session State",
    priority: "High",
    icon: "🔋",
    color: "#8b5cf6",
    description: "Real-time estimate of the patient's current cognitive and motor fatigue level",
    details: [
      "Minutes since conversation started — longer = more distortion",
      "Speech rate trend: declining words-per-minute indicates fatigue",
      "Error density trend: increasing distortion rate over time",
      "Pause frequency and duration trend: longer pauses = more motor planning difficulty",
      "Signal energy decline: quieter speech often correlates with fatigue",
      "Current fatigue phase classification: fresh / mild / moderate / severe",
    ],
    whyLLM: "A patient's confusion matrix at minute 2 is different from minute 20. The LLM needs to dynamically adjust how aggressively it corrects. Early in a conversation, a /d/ might actually be /d/. Twenty minutes in, it's almost certainly an attempted /θ/. Without fatigue tracking, the model either over-corrects fresh speech or under-corrects fatigued speech.",
    example: "Session state: 18 minutes elapsed, fatigue phase: moderate\n→ Error density has increased 40% from baseline\n→ LLM shifts confusion matrix weights: P(/θ/ | /d/) increases from 0.70 → 0.85\n→ LLM also increases tolerance for missing syllables and word fragments",
  },
  {
    id: "conversation-context",
    title: "Conversational Context Window",
    priority: "High",
    icon: "💬",
    color: "#3b82f6",
    description: "What has been said so far in this conversation — by the patient AND by others",
    details: [
      "Last N translated utterances from the patient (with confidence scores)",
      "Responses from the listener/caregiver/doctor (if captured or typed in)",
      "Topic tracking: what subjects have been discussed",
      "Unresolved questions: things the patient was asked but hasn't answered yet",
      "Emotional trajectory: escalating frustration, calm, distress signals",
      "Repeated attempts: when the patient is trying to say the same thing again",
    ],
    whyLLM: "Context is the LLM's most powerful disambiguation tool. If a doctor asks 'where does it hurt?' and the patient's next utterance is ambiguous, the LLM knows the response is likely a body part. If the patient has said 'water' three times and the listener hasn't responded, the next attempt — however garbled — is probably 'water' again. Without conversation context, every utterance is interpreted in isolation.",
    example: "Nurse: 'Are you feeling any pain?'\nPatient (translated): 'Yes my head'\nNurse: 'On a scale of 1 to 10?'\nPatient (raw): '[ambiguous sounds]'\n→ Context tells LLM: expect a number between 1-10\n→ Resolves ambiguous phonemes toward number words",
  },
  {
    id: "situational-context",
    title: "Situational / Environmental Context",
    priority: "Medium",
    icon: "📍",
    color: "#10b981",
    description: "Where the patient is, who they're talking to, and what activity is happening",
    details: [
      "Setting: hospital room, therapy session, home, family visit, mealtime",
      "Listener identity: doctor, nurse, family member, therapist, stranger",
      "Time of day: morning (typically clearer speech) vs. evening (more fatigue)",
      "Activity context: medication time, physical therapy, eating, resting",
      "Recent medical events: surgery, medication changes, therapy milestones",
      "Communication urgency: routine conversation vs. expressing pain/emergency",
    ],
    whyLLM: "Situational context dramatically narrows the vocabulary the LLM expects. During mealtime, food/drink words get weighted higher. In a medical exam, body parts and symptom descriptions rise. With family, emotional and personal vocabulary increases. This Bayesian prior is the difference between 'I can't tell what they said' and 'given it's mealtime, this is almost certainly about food.'",
    example: "Context: mealtime, family visit\nPatient (raw): 'pah duh [pause] sawt'\n→ Without context: 'pod the salt'? 'pod the sought'? 'pass the salt'?\n→ With mealtime context: 'pass the salt' (confidence: 92%)",
  },
  {
    id: "patient-vocab",
    title: "Patient Personal Vocabulary",
    priority: "Medium",
    icon: "📖",
    color: "#06b6d4",
    description: "Words, names, and phrases specific to this patient's life that a generic model wouldn't know",
    details: [
      "Family member names and nicknames",
      "Pet names, place names, home locations",
      "Medication names the patient takes regularly",
      "Frequently used personal phrases and expressions",
      "Preferred terms for needs (some patients say 'potty' not 'bathroom')",
      "Cultural/language background: borrowed words from other languages",
    ],
    whyLLM: "The LLM's language model assigns near-zero probability to proper names it's never seen. If the patient is trying to say their granddaughter 'Adalynn' and the LLM doesn't have that name, it might resolve to 'add a line' or 'Adeline'. The personal vocabulary acts as a boosted lexicon — these words get elevated probability during beam search.",
    example: "Personal vocab: dog named 'Biscuit', daughter 'Priya', takes 'Lisinopril'\nPatient (raw): 'where ih [pause] bih-kit'\n→ Without personal vocab: 'where is bucket?' (best guess)\n→ With personal vocab: 'where is Biscuit?' (matches pet name + phoneme pattern)",
  },
];

const PROCESSING_FEATURES = [
  {
    id: "beam-search",
    title: "Constrained Beam Search Decoding",
    icon: "🔍",
    color: "#ef4444",
    description: "Generate multiple candidate sentences simultaneously, scored against all available signals",
    mechanism: [
      "Start with phoneme probability sequence as input",
      "Generate top-K candidate word sequences using language model probability",
      "Score each candidate against the patient's distortion model (does this word map to those distorted phonemes?)",
      "Score against conversational context (does this make sense as a response?)",
      "Score against situational context (is this vocabulary expected right now?)",
      "Rank candidates by combined score, surface top-3 with confidence",
    ],
    edgeCase: "When top candidates are close in score (e.g., 'water' at 0.45 vs. 'daughter' at 0.40), the system should surface both rather than guess. The listener can tap the correct one, which feeds back into the model.",
  },
  {
    id: "word-recovery",
    title: "Missing Word & Syllable Recovery",
    icon: "🧩",
    color: "#f59e0b",
    description: "Reconstruct words and grammatical elements the patient's motor system dropped",
    mechanism: [
      "Detect gaps in the phoneme sequence that suggest a dropped word (based on timing/prosody)",
      "Use grammatical expectations to identify what's missing ('I want go bathroom' → 'to' is missing)",
      "Apply patient's syllable deletion profile (if they drop first syllables, 'nana' might be 'banana')",
      "Distinguish intentional pauses from motor-failure gaps using timing signatures",
      "Handle function word omission (articles, prepositions, pronouns) which is common in agrammatic speech",
    ],
    edgeCase: "The system must differentiate between a patient who has agrammatic aphasia (genuinely telegraphic speech: 'want water now') and a patient whose motor system dropped the function words. The former should be translated as-is; the latter should have words recovered. Calibration data helps classify this.",
  },
  {
    id: "repeat-detection",
    title: "Repetition & Self-Correction Detection",
    icon: "🔄",
    color: "#8b5cf6",
    description: "Recognize when the patient is retrying a word or self-correcting, not saying new content",
    mechanism: [
      "Detect phoneme sequences that partially overlap with the immediately prior attempt",
      "Identify 'groping' patterns: multiple starts on the same word (t- t- te- tele- telephone)",
      "Distinguish self-correction (patient replacing a wrong word) from perseveration (involuntary repetition)",
      "Identify when the patient abandons a word attempt and substitutes a simpler alternative",
      "Collapse retry sequences into a single output word using the best attempt",
    ],
    edgeCase: "Perseveration (involuntary repetition of a word or sound) is common in frontal lobe injuries. 'I want want want water water' — the patient means 'I want water' but cannot stop repeating. The system needs to recognize perseveration patterns from calibration and filter them, but NOT filter intentional emphasis ('I really really need help').",
  },
  {
    id: "confidence-calibration",
    title: "Confidence Calibration & Uncertainty Signaling",
    icon: "📊",
    color: "#3b82f6",
    description: "Accurately estimate HOW CERTAIN the translation is, and communicate uncertainty honestly",
    mechanism: [
      "Compute per-word confidence from phoneme probability alignment strength",
      "Compute per-sentence confidence from semantic coherence score",
      "Calibrate confidence against historical accuracy (if model says 90% confident, it should be right 90% of the time)",
      "Flag specific words that drove down confidence (highlight uncertain words visually)",
      "Detect when total confidence is too low to display any translation (show 'I couldn't understand, please try again')",
      "Track and present confidence trends across the conversation",
    ],
    edgeCase: "Over-confidence is more dangerous than under-confidence. If the system shows 'I am in pain' with high confidence when the patient said something completely different, medical staff might act on wrong information. The system should err toward showing uncertainty and multiple options rather than committing to a wrong translation.",
  },
  {
    id: "emergency-detect",
    title: "Emergency & Urgency Detection",
    icon: "🚨",
    color: "#dc2626",
    description: "Prioritize and fast-track utterances that signal pain, danger, or urgent medical needs",
    mechanism: [
      "Maintain a priority vocabulary: pain, help, hurt, can't breathe, falling, emergency, choking",
      "Apply lower confidence thresholds for emergency words (surface them even at 50% confidence)",
      "Detect prosodic urgency cues: sudden volume increase, pitch changes, rapid speech attempts",
      "Trigger visual/audio alerts on the listener's display when emergency content is detected",
      "Emergency phrases bypass the normal beam search and use a fast-path detection model",
    ],
    edgeCase: "False positives on emergency detection are acceptable — it's better to alert a nurse unnecessarily than to miss a patient in distress. But false positives should be trackable so the system can learn what this patient's non-emergency speech sounds like when it resembles emergency phrases.",
  },
  {
    id: "temporal-adaptation",
    title: "Temporal Model Adaptation",
    icon: "⏱️",
    color: "#10b981",
    description: "Adjust the distortion model in real-time based on how the patient's speech changes during a session",
    mechanism: [
      "Continuously update distortion probabilities based on confirmed translations during the session",
      "Track fatigue curve: map error density against elapsed time to predict upcoming degradation",
      "Detect sudden changes: if speech abruptly worsens, flag possible medical event (new stroke, seizure, medication reaction)",
      "Adapt to therapy gains: if patient is improving over weeks, relax correction aggressiveness gradually",
      "Time-of-day modeling: learn that this patient speaks more clearly at 10am than 8pm",
    ],
    edgeCase: "A sudden, dramatic change in speech pattern mid-conversation could indicate a new neurological event. The system should distinguish 'gradual fatigue' (normal) from 'sudden cliff' (potentially dangerous) and alert caregivers if it detects the latter.",
  },
];

const OUTPUT_FEATURES = [
  {
    id: "primary-translation",
    title: "Primary Translation with Word-Level Confidence",
    description: "The most likely intended sentence, with each word color-coded by confidence",
  },
  {
    id: "alternatives",
    title: "Alternative Interpretations (Top-3)",
    description: "When primary confidence is below threshold, show 2-3 alternatives the listener can choose from",
  },
  {
    id: "uncertainty-markers",
    title: "Uncertain Word Highlighting",
    description: "Visually mark which specific words in the translation are uncertain — the rest can be trusted",
  },
  {
    id: "emergency-flag",
    title: "Emergency Content Flag",
    description: "Prominent visual + optional audio alert when pain, distress, or urgent medical content is detected",
  },
  {
    id: "context-hint",
    title: "Context Hint for Listener",
    description: "Brief note when context heavily influenced the translation: 'Interpreted in context of mealtime conversation'",
  },
  {
    id: "retry-prompt",
    title: "Intelligent Retry Suggestions",
    description: "When confidence is too low, suggest specific words for the patient to try again: 'Can you try saying the first word again?'",
  },
  {
    id: "feedback-capture",
    title: "Listener Feedback Capture",
    description: "Tap-to-confirm or tap-to-correct on the translated text — feeds directly back into the distortion model",
  },
];

export default function LLMFeatureSpec() {
  const [activeTab, setActiveTab] = useState("inputs");
  const [expandedItem, setExpandedItem] = useState(null);

  const colors = {
    bg: "#0b0d13",
    surface: "#131620",
    surfaceHover: "#1a1e2c",
    text: "#e0e3eb",
    textMuted: "#646b8a",
    border: "#1f2438",
    cardBg: "#10121c",
  };

  const priorityColors = {
    Critical: { bg: "rgba(239,68,68,0.1)", text: "#ef4444", border: "rgba(239,68,68,0.25)" },
    High: { bg: "rgba(139,92,246,0.1)", text: "#8b5cf6", border: "rgba(139,92,246,0.25)" },
    Medium: { bg: "rgba(16,185,129,0.1)", text: "#10b981", border: "rgba(16,185,129,0.25)" },
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: colors.bg,
      color: colors.text,
      fontFamily: "'IBM Plex Sans', 'DM Sans', system-ui, sans-serif",
    }}>
      <div style={{ maxWidth: 850, margin: "0 auto", padding: "20px 20px 60px" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 8 }}>
          <span style={{ fontSize: 22 }}>✨</span>
          <span style={{ fontSize: 17, fontWeight: 700, letterSpacing: "-0.02em" }}>
            LLM Layer — Feature Specification
          </span>
        </div>
        <p style={{ color: colors.textMuted, fontSize: 13, lineHeight: 1.5, marginBottom: 24 }}>
          Everything the LLM receives, processes, and outputs to translate brain injury patient speech.
        </p>

        {/* Tabs */}
        <div style={{
          display: "flex",
          gap: 4,
          marginBottom: 28,
          background: colors.surface,
          borderRadius: 10,
          padding: 4,
          border: `1px solid ${colors.border}`,
        }}>
          {[
            { key: "inputs", label: "Inputs (What LLM Receives)", count: INPUT_FEATURES.length },
            { key: "processing", label: "Processing (What LLM Does)", count: PROCESSING_FEATURES.length },
            { key: "outputs", label: "Outputs (What LLM Returns)", count: OUTPUT_FEATURES.length },
          ].map(tab => (
            <button
              key={tab.key}
              onClick={() => { setActiveTab(tab.key); setExpandedItem(null); }}
              style={{
                flex: 1,
                padding: "10px 12px",
                borderRadius: 8,
                border: "none",
                background: activeTab === tab.key ? "#1f2438" : "transparent",
                color: activeTab === tab.key ? colors.text : colors.textMuted,
                fontSize: 12,
                fontWeight: 600,
                cursor: "pointer",
                transition: "all 0.2s",
              }}
            >
              {tab.label} ({tab.count})
            </button>
          ))}
        </div>

        {/* ═══ INPUTS TAB ═══ */}
        {activeTab === "inputs" && (
          <div>
            <div style={{
              display: "flex",
              gap: 8,
              marginBottom: 20,
              flexWrap: "wrap",
            }}>
              {["Critical", "High", "Medium"].map(p => (
                <span key={p} style={{
                  fontSize: 11,
                  fontWeight: 600,
                  padding: "4px 10px",
                  borderRadius: 6,
                  background: priorityColors[p].bg,
                  color: priorityColors[p].text,
                  border: `1px solid ${priorityColors[p].border}`,
                }}>
                  {p} — {INPUT_FEATURES.filter(f => f.priority === p).length} features
                </span>
              ))}
            </div>

            {INPUT_FEATURES.map((feature) => {
              const isExpanded = expandedItem === feature.id;
              const pc = priorityColors[feature.priority];
              return (
                <div key={feature.id} style={{ marginBottom: 12 }}>
                  <div
                    style={{
                      background: isExpanded ? colors.surfaceHover : colors.cardBg,
                      border: `1px solid ${isExpanded ? feature.color + "40" : colors.border}`,
                      borderRadius: 14,
                      overflow: "hidden",
                      cursor: "pointer",
                      transition: "all 0.25s",
                      boxShadow: isExpanded ? `0 4px 20px ${feature.color}10` : "none",
                    }}
                    onClick={() => setExpandedItem(isExpanded ? null : feature.id)}
                  >
                    <div style={{ padding: "16px 20px", display: "flex", alignItems: "flex-start", gap: 14 }}>
                      <span style={{ fontSize: 22, flexShrink: 0, marginTop: 2 }}>{feature.icon}</span>
                      <div style={{ flex: 1 }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 8, flexWrap: "wrap", marginBottom: 4 }}>
                          <span style={{ fontSize: 15, fontWeight: 600 }}>{feature.title}</span>
                          <span style={{
                            fontSize: 10,
                            fontWeight: 700,
                            padding: "2px 8px",
                            borderRadius: 4,
                            background: pc.bg,
                            color: pc.text,
                            border: `1px solid ${pc.border}`,
                          }}>
                            {feature.priority}
                          </span>
                        </div>
                        <div style={{ color: colors.textMuted, fontSize: 13, lineHeight: 1.4 }}>
                          {feature.description}
                        </div>
                      </div>
                      <div style={{
                        color: colors.textMuted,
                        fontSize: 18,
                        transform: isExpanded ? "rotate(180deg)" : "rotate(0)",
                        transition: "transform 0.2s",
                        flexShrink: 0,
                        marginTop: 4,
                      }}>▾</div>
                    </div>

                    {isExpanded && (
                      <div style={{ padding: "0 20px 20px", borderTop: `1px solid ${colors.border}` }}>
                        <div style={{ paddingTop: 16 }}>
                          <div style={{ fontSize: 11, fontWeight: 600, color: feature.color, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10 }}>
                            What this contains
                          </div>
                          {feature.details.map((d, i) => (
                            <div key={i} style={{ display: "flex", gap: 10, marginBottom: 7, fontSize: 13, lineHeight: 1.5 }}>
                              <span style={{ color: feature.color, flexShrink: 0 }}>●</span>
                              <span>{d}</span>
                            </div>
                          ))}
                        </div>

                        <div style={{
                          marginTop: 16,
                          padding: 14,
                          background: `${feature.color}10`,
                          borderRadius: 10,
                          border: `1px solid ${feature.color}20`,
                        }}>
                          <div style={{ fontSize: 11, fontWeight: 600, color: feature.color, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>
                            Why the LLM needs this
                          </div>
                          <div style={{ fontSize: 13, lineHeight: 1.6 }}>{feature.whyLLM}</div>
                        </div>

                        <div style={{
                          marginTop: 12,
                          padding: 14,
                          background: colors.surface,
                          borderRadius: 10,
                          border: `1px solid ${colors.border}`,
                          fontFamily: "'IBM Plex Mono', 'Fira Code', monospace",
                          fontSize: 12,
                          lineHeight: 1.7,
                          whiteSpace: "pre-wrap",
                          color: colors.textMuted,
                        }}>
                          <div style={{ fontSize: 11, fontWeight: 600, color: colors.text, fontFamily: "'IBM Plex Sans', system-ui", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 8 }}>
                            Example
                          </div>
                          {feature.example}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* ═══ PROCESSING TAB ═══ */}
        {activeTab === "processing" && (
          <div>
            {PROCESSING_FEATURES.map((feature) => {
              const isExpanded = expandedItem === feature.id;
              return (
                <div key={feature.id} style={{ marginBottom: 12 }}>
                  <div
                    style={{
                      background: isExpanded ? colors.surfaceHover : colors.cardBg,
                      border: `1px solid ${isExpanded ? feature.color + "40" : colors.border}`,
                      borderRadius: 14,
                      overflow: "hidden",
                      cursor: "pointer",
                      transition: "all 0.25s",
                    }}
                    onClick={() => setExpandedItem(isExpanded ? null : feature.id)}
                  >
                    <div style={{ padding: "16px 20px", display: "flex", alignItems: "flex-start", gap: 14 }}>
                      <span style={{ fontSize: 22, flexShrink: 0, marginTop: 2 }}>{feature.icon}</span>
                      <div style={{ flex: 1 }}>
                        <span style={{ fontSize: 15, fontWeight: 600 }}>{feature.title}</span>
                        <div style={{ color: colors.textMuted, fontSize: 13, marginTop: 4, lineHeight: 1.4 }}>
                          {feature.description}
                        </div>
                      </div>
                      <div style={{
                        color: colors.textMuted, fontSize: 18,
                        transform: isExpanded ? "rotate(180deg)" : "rotate(0)",
                        transition: "transform 0.2s", flexShrink: 0, marginTop: 4,
                      }}>▾</div>
                    </div>

                    {isExpanded && (
                      <div style={{ padding: "0 20px 20px", borderTop: `1px solid ${colors.border}` }}>
                        <div style={{ paddingTop: 16 }}>
                          <div style={{ fontSize: 11, fontWeight: 600, color: feature.color, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 10 }}>
                            Mechanism
                          </div>
                          {feature.mechanism.map((m, i) => (
                            <div key={i} style={{ display: "flex", gap: 10, marginBottom: 7, fontSize: 13, lineHeight: 1.5 }}>
                              <span style={{
                                color: feature.color, flexShrink: 0,
                                fontSize: 11, fontWeight: 700,
                                width: 20, textAlign: "right",
                              }}>{i + 1}.</span>
                              <span>{m}</span>
                            </div>
                          ))}
                        </div>

                        <div style={{
                          marginTop: 16,
                          padding: 14,
                          background: "rgba(245,158,11,0.06)",
                          borderRadius: 10,
                          border: "1px solid rgba(245,158,11,0.15)",
                        }}>
                          <div style={{ fontSize: 11, fontWeight: 600, color: "#f59e0b", textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>
                            Critical Edge Case
                          </div>
                          <div style={{ fontSize: 13, lineHeight: 1.6 }}>{feature.edgeCase}</div>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* ═══ OUTPUTS TAB ═══ */}
        {activeTab === "outputs" && (
          <div>
            <p style={{ color: colors.textMuted, fontSize: 13, lineHeight: 1.5, marginBottom: 20 }}>
              The LLM returns a structured response for every utterance. The display layer
              renders these fields into the listener-facing interface.
            </p>

            {OUTPUT_FEATURES.map((feature, idx) => (
              <div key={feature.id} style={{
                background: colors.cardBg,
                border: `1px solid ${colors.border}`,
                borderRadius: 12,
                padding: "14px 18px",
                marginBottom: 10,
                display: "flex",
                gap: 14,
                alignItems: "flex-start",
              }}>
                <div style={{
                  width: 28, height: 28, borderRadius: 7,
                  background: "rgba(59,130,246,0.1)",
                  border: "1px solid rgba(59,130,246,0.2)",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: 12, fontWeight: 700, color: "#3b82f6", flexShrink: 0,
                }}>
                  {idx + 1}
                </div>
                <div>
                  <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 4 }}>{feature.title}</div>
                  <div style={{ fontSize: 13, color: colors.textMuted, lineHeight: 1.5 }}>{feature.description}</div>
                </div>
              </div>
            ))}

            <div style={{
              marginTop: 24,
              padding: 20,
              background: colors.surface,
              borderRadius: 14,
              border: `1px solid ${colors.border}`,
            }}>
              <div style={{ fontSize: 14, fontWeight: 600, marginBottom: 14 }}>Example LLM Response Structure</div>
              <pre style={{
                fontFamily: "'IBM Plex Mono', 'Fira Code', monospace",
                fontSize: 12,
                lineHeight: 1.7,
                color: colors.textMuted,
                whiteSpace: "pre-wrap",
                margin: 0,
                padding: 16,
                background: colors.bg,
                borderRadius: 10,
                border: `1px solid ${colors.border}`,
              }}>{`{
  "primary": {
    "text": "I think I need some water",
    "confidence": 0.87,
    "word_confidence": [
      {"word": "I",     "conf": 0.98},
      {"word": "think", "conf": 0.72, "flag": "uncertain"},
      {"word": "I",     "conf": 0.98},
      {"word": "need",  "conf": 0.85},
      {"word": "some",  "conf": 0.91},
      {"word": "water", "conf": 0.94}
    ]
  },
  "alternatives": [
    {"text": "I think I need some water",   "conf": 0.87},
    {"text": "I think I need the water",    "conf": 0.63},
    {"text": "I drink I need some water",   "conf": 0.41}
  ],
  "emergency_flag": false,
  "context_note": null,
  "fatigue_phase": "mild",
  "retry_suggestion": null,
  "raw_phonemes": "aɪ d(0.42)θ(0.30) ɪŋk aɪ niː sʌm wɑːɚ"
}`}</pre>
            </div>

            <div style={{
              marginTop: 20,
              padding: 20,
              background: "rgba(239,68,68,0.06)",
              borderRadius: 14,
              border: "1px solid rgba(239,68,68,0.15)",
            }}>
              <div style={{ fontSize: 14, fontWeight: 600, color: "#ef4444", marginBottom: 8 }}>
                Critical Design Principle: Transparency Over Confidence
              </div>
              <div style={{ fontSize: 13, color: colors.textMuted, lineHeight: 1.7 }}>
                In medical settings, a confident wrong translation is far more dangerous than an honest "I'm not sure."
                The system should NEVER hide uncertainty to appear smoother. If the model is 50% confident,
                the listener needs to know that — they'll ask the patient to repeat, or use a different
                communication method. The output structure is designed to make uncertainty visible,
                actionable, and non-threatening. A yellow-highlighted word is an invitation to clarify,
                not a failure state.
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
