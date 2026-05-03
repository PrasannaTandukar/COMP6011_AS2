OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:32b"

LABELS = ["safe", "indicator", "ideation", "behavior", "attempt"]

RISK_DEFINITIONS = """
You must classify the conversation into exactly one of the following suicide risk levels:

safe:
The conversation discusses general life topics, wellbeing, or mental health without immediate suicide risk.

indicator:
The user shows severe distress signals such as hopelessness, burdensomeness, isolation, feeling like a failure, or wanting to disappear, but does not clearly express suicidal thoughts or intent.

ideation:
The user expresses thoughts of dying, wanting life or pain to end, or wanting not to exist, but there is no clear immediate plan, preparation, or active attempt.

behavior:
The user describes actions, preparations, timing, escalating intent, self-isolation, acquiring means, or other behaviours suggesting increased suicide risk.

attempt:
The user discusses past attempts, active self-harm, imminent plans, clear intent, or immediate danger.
"""

RISK_ORDER = {
    "safe": 0,
    "indicator": 1,
    "ideation": 2,
    "behavior": 3,
    "attempt": 4
}

REVERSE_RISK_ORDER = {v: k for k, v in RISK_ORDER.items()}
