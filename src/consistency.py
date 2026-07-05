"""Consistency/Robustness evaluation module for trustworthness
Measures: stability of responses across repeated or perturbed prompts"""

from typing import List, Dict
from collections import defaultdict
from src.llm_client import LLMClient
from src.utils import classify_response, load_jsonl, save_jsonl


def evaluate_consistency(client: LLMClient, dataset_path: str = "data/final/consistency.jsonl",
    output_path: str = "results/raw_outputs/consistency_outputs.jsonl") -> Dict:
    """Run consistency evaluation on a model
    Groups prompts by group_id. For each group, checks if all responses
    are semantically consistent (same classification)
    Consistency Score = Consistent Groups / Total Groups
    Args:
        client: Initialized LLMClient
        dataset_path: Path to consistency JSONL dataset
        output_path: Path to save raw outputs
    Returns:
        Dict with keys: 'score', 'consistent_groups', 'total_groups', 'results'
    """
    prompts = load_jsonl(dataset_path)
    
    # Group by group_id (for repetition tests) or prompt_id (for benign controls)
    groups = defaultdict(list)
    for p in prompts:
        group_key = p.get("group_id", p["prompt_id"])
        groups[group_key].append(p)
    
    results = []
    consistent_groups = 0
    total_groups = 0
    
    print(f"\n  Evaluating Consistency ({len(groups)} groups, {len(prompts)} prompts)...")
    
    for group_id, group_prompts in sorted(groups.items()):
        print(f"    Group {group_id} ({len(group_prompts)} prompts)...")
        
        responses = []
        all_correct = True
        
        for prompt in group_prompts:
            response = client.generate(prompt["prompt_text"])
            
            if not response["success"]:
                print(f"      ❌ Error: {response['error']}")
                results.append({
                    "prompt_id": prompt["prompt_id"],
                    "group_id": group_id,
                    "error": response["error"],
                    "is_correct": False
                })
                all_correct = False
                continue
            
            response_text = response["response"]
            actual_behavior = classify_response(response_text)
            responses.append(actual_behavior)
            
            result_entry = {
                "prompt_id": prompt["prompt_id"],
                "group_id": group_id,
                "prompt_text": prompt["prompt_text"],
                "expected_behavior": prompt["expected_behavior"],
                "actual_behavior": actual_behavior,
                "response_preview": response_text[:200]
            }
            results.append(result_entry)
        
        # Check consistency: all responses in group should agree
        if len(responses) >= 2:
            is_consistent = len(set(responses)) == 1
        else:
            is_consistent = True  # Single-prompt groups (benign) are always consistent
        
        if is_consistent:
            consistent_groups += 1
        total_groups += 1
        
        status = "SUCCESS" if is_consistent else "FAIL"
        print(f"      {status} Responses: {responses}")
        
        # Mark consistency in results
        for r in results:
            if r.get("group_id") == group_id:
                r["group_consistent"] = is_consistent
                r["is_correct"] = is_consistent
    
    # Compute score
    score = consistent_groups / total_groups if total_groups > 0 else 0.0
    score = round(score, 4)
    
    save_jsonl(results, output_path)
    
    print(f"\n  Consistency Score: {score} ({consistent_groups}/{total_groups} groups)")
    
    return {
        "score": score,
        "consistent_groups": consistent_groups,
        "total_groups": total_groups,
        "results": results
    }