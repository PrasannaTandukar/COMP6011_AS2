import requests
import json
import re
import constants
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Call the local ollama model
def call_ollama(prompt, model, url, temperature=0):
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }

    response = requests.post(url, json=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(f"Ollama error {response.status_code}: {response.text}")

    return response.json()["response"]


def build_llm_only_prompt(dialogue, risk_definitions):
    return f"""
You are a suicide-risk screening assistant for research purposes.

Important constraints:
- You are not making a clinical diagnosis.
- You are classifying the suicide risk level in a conversation.
- Use only the evidence in the conversation.
- Do not invent facts.
- Return valid JSON only.
- Choose exactly one label from: safe, indicator, ideation, behavior, attempt.

Risk level definitions:
{risk_definitions}

Conversation:
\"\"\"
{dialogue}
\"\"\"

Return your answer in this JSON format only:
{{
  "risk_level": "one of: safe, indicator, ideation, behavior, attempt",
  "confidence": "low, medium, or high",
  "evidence": "brief evidence from the conversation",
  "reason": "brief explanation for the selected label"
}}
"""

#    Attempts to extract JSON from LLM response.
#    Handles cases where the model adds extra text.
def extract_json_from_response(response_text):
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "risk_level": "parse_error",
        "confidence": "unknown",
        "evidence": "",
        "reason": response_text[:500]
    }

def predict_llm_only(dialogue):
    prompt = build_llm_only_prompt(dialogue, constants.RISK_DEFINITIONS)
    response = call_ollama(prompt, constants.MODEL_NAME, constants.OLLAMA_URL)
    parsed = extract_json_from_response(response)

    pred = str(parsed.get("risk_level", "parse_error")).strip().lower()

    if pred not in constants.LABELS:
        pred = "parse_error"

    return {
        "prediction": pred,
        "confidence": parsed.get("confidence", ""),
        "evidence": parsed.get("evidence", ""),
        "reason": parsed.get("reason", ""),
        "raw_response": response
    }

# Simple deterministic safety rules.
def safety_rule_layer(dialogue):
    text = dialogue.lower()

    attempt_patterns = [
        r"\bi'?ll be gone\b",
        r"\bby october\b",
        r"\bend of the road\b",
        r"\bif i actually did what i.?ve been thinking\b",
        r"\bdrive off the bridge\b",
        r"\brope around my neck\b",
        r"\bactive self-harm\b",
        r"\bkill myself\b",
        r"\bend it tonight\b",
        r"\bi have a plan\b",
        r"\bi.?m going to do it\b"
    ]

    behavior_patterns = [
        r"\bprepar",
        r"\bacquiring means\b",
        r"\bwait until\b",
        r"\bwhen the season changes\b",
        r"\bno path forward\b",
        r"\bdelete(d)?\b",
        r"\bdisappear\b",
        r"\bi.?ll be gone\b"
    ]

    ideation_patterns = [
        r"\bi want to die\b",
        r"\bstill want to die\b",
        r"\bwant the pain to end\b",
        r"\bwhy am i even here\b",
        r"\bno point\b",
        r"\beverything ends anyway\b",
        r"\bwaiting for the clock to run out\b"
    ]

    indicator_patterns = [
        r"\bhopeless\b",
        r"\bburden\b",
        r"\bfailure\b",
        r"\bdisposable\b",
        r"\breplaceable\b",
        r"\bno future\b",
        r"\bempty\b",
        r"\bworthless\b",
        r"\bcompletely alone\b"
    ]

    for pattern in attempt_patterns:
        if re.search(pattern, text):
            return {
                "rule_label": "attempt",
                "rule_trigger": pattern
            }

    for pattern in behavior_patterns:
        if re.search(pattern, text):
            return {
                "rule_label": "behavior",
                "rule_trigger": pattern
            }

    for pattern in ideation_patterns:
        if re.search(pattern, text):
            return {
                "rule_label": "ideation",
                "rule_trigger": pattern
            }

    for pattern in indicator_patterns:
        if re.search(pattern, text):
            return {
                "rule_label": "indicator",
                "rule_trigger": pattern
            }

    return {
        "rule_label": "safe",
        "rule_trigger": ""
    }

#    If the safety rule identifies higher risk than the LLM, use the higher-risk label.
def hybrid_decision(llm_prediction, rule_prediction):
    llm_score = constants.RISK_ORDER.get(llm_prediction, -1)
    rule_score = constants.RISK_ORDER.get(rule_prediction, -1)

    final_score = max(llm_score, rule_score)

    if final_score == -1:
        return "parse_error"

    # converts the number back into the text label
    return constants.REVERSE_RISK_ORDER[final_score]

def predict_hybrid(dialogue):
    llm_result = predict_llm_only(dialogue)
    rule_result = safety_rule_layer(dialogue)

    final_prediction = hybrid_decision(
        llm_result["prediction"],
        rule_result["rule_label"]
    )

    return {
        "prediction": final_prediction,
        "llm_prediction": llm_result["prediction"],
        "rule_prediction": rule_result["rule_label"],
        "rule_trigger": rule_result["rule_trigger"],
        "confidence": llm_result["confidence"],
        "evidence": llm_result["evidence"],
        "reason": llm_result["reason"],
        "raw_response": llm_result["raw_response"]
    }

def evaluate_predictions(results_df, model_name):
    y_true = results_df["Actual"]
    y_pred = results_df["Predicted"]

    valid_mask = y_pred.isin(constants.LABELS)

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    print("=" * 80)
    print(model_name)
    print("=" * 80)

    print(f"Total cases: {len(results_df)}")
    print(f"Valid predictions: {valid_mask.sum()}")
    print(f"Parse errors: {(~valid_mask).sum()}")

    accuracy = accuracy_score(y_true_valid, y_pred_valid)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_valid,
        y_pred_valid,
        labels=constants.LABELS,
        average="macro",
        zero_division=0
    )

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro Precision: {precision:.3f}")
    print(f"Macro Recall: {recall:.3f}")
    print(f"Macro F1: {f1:.3f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true_valid,
        y_pred_valid,
        labels=constants.LABELS,
        zero_division=0
    ))

    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=constants.LABELS)

    return {
        "model": model_name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "confusion_matrix": cm
    }

def plot_confusion_matrix(cm, labels, title):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(title)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.show()

def show_errors(results_df, model_name):
    errors = results_df[results_df["Actual"] != results_df["Predicted"]]

    print("=" * 80)
    print(f"Errors for {model_name}")
    print("=" * 80)

    if len(errors) == 0:
        print("No errors.")
        return errors

    for _, row in errors.iterrows():
        print(f"\nCase ID: {row['Case ID']}")
        print(f"Actual: {row['Actual']}")
        print(f"Predicted: {row['Predicted']}")
        print(f"Evidence: {row.get('Evidence', '')}")
        print(f"Reason: {row.get('Reason', '')}")

    return errors