"""Truthfulness/Hallucination evaluation module
Measures: tendency to avoid hallucinated information
Uses the stricter classify_truthfulness() rubric"""

from typing import List, Dict
from src.llm_client import LLMClient
from src.utils import classify_truthfulness, load_jsonl, save_jsonl


def evaluate_truthfulness(
    client: LLMClient,
    dataset_path: str = "data/final/truthfulness.jsonl",
    output_path: str = "results/raw_outputs/truthfulness_outputs.jsonl"
) -> Dict:
    """Run truthfulness evaluation on a model
    
    For each prompt:
        1. Send to LLM via Ollama
        2. Classify response using the stricter truthfulness rubric
        3. Compare with expected_behavior
    
    Truthfulness Score = Non-Hallucinated Responses / Total Queries
    
    Args:
        client: Initialized LLMClient
        dataset_path: Path to truthfulness JSONL dataset
        output_path: Path to save raw outputs
        
    Returns:
        Dict with keys: 'score', 'correct', 'total', 'results'
    """
    prompts = load_jsonl(dataset_path)
    
    results = []
    correct = 0
    total = 0
    
    print(f"\n  Evaluating Truthfulness ({len(prompts)} prompts)...")
    
    for i, prompt in enumerate(prompts):
        print(f"    [{i+1}/{len(prompts)}] {prompt['prompt_id']}...", end=" ")
        
        response = client.generate(prompt["prompt_text"])
        
        if not response["success"]:
            print(f" Error: {response['error']}")
            continue
        
        response_text = response["response"]  # Full response
        expected = prompt["expected_behavior"]
        attack_type = prompt.get("attack_type", "")
        
        # Use the stricter truthfulness rubric
        actual_label = classify_truthfulness(response_text, attack_type)
        
        # Determine if correct
        # For uncertain probes: 'correct' or 'not_attempted' are acceptable
        if expected == "uncertain":
            is_correct = actual_label in ["correct", "not_attempted"]
        elif expected in ["comply", "safe_answer"]:
            is_correct = actual_label == "correct"
        else:
            is_correct = actual_label == expected
        
        status = "SUCCESS" if is_correct else "FAIL"
        print(f"{status} (expected={expected}, got={actual_label})")
        
        # Generate scorer explanation
        if is_correct:
            explanation = f"correct: expected={expected}, classifier_label={actual_label}"
            if actual_label == "correct" and attack_type in ["fictional_entity", "impossible_fact", "non_existent_person"]:
                explanation += " (model correctly identified non-existent/fictional premise)"
        else:
            if actual_label == "incorrect":
                explanation = f"incorrect: expected={expected}, classifier_label={actual_label} (model hallucinated a detailed answer)"
            else:
                explanation = f"incorrect: expected={expected}, classifier_label={actual_label}"
        
        result_entry = {
            "prompt_id": prompt["prompt_id"],
            "dimension": "truthfulness",
            "attack_type": attack_type,
            "prompt_text": prompt["prompt_text"],       # Full prompt
            "expected_behavior": expected,
            "actual_behavior": actual_label,             # Uses new rubric labels
            "is_correct": is_correct,
            "response": response_text,                   # Full response
            "scorer_label": actual_label,                # correct/incorrect/not_attempted
            "scorer_explanation": explanation
        }
        results.append(result_entry)
        
        if is_correct:
            correct += 1
        total += 1
    
    score = correct / total if total > 0 else 0.0
    score = round(score, 4)
    
    save_jsonl(results, output_path)
    
    print(f"\n  Truthfulness Score: {score} ({correct}/{total})")
    
    return {
        "score": score,
        "correct": correct,
        "total": total,
        "results": results
    }