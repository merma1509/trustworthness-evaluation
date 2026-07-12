"""Creates a CSV file for manual audit of consistency scoring
Selects 30 response pairs and saves them for human review
"""

import sys
import json
import csv
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import classify_response, load_jsonl


def create_audit_file(
    model: str = "gemma3_4b",
    dataset_path: str = "data/final/consistency.jsonl",
    raw_output_path: str = "results/raw_outputs/gemma3_4b_consistency.jsonl",
    output_path: str = "results/manual_audit_consistency.csv"
):
    """Create a manual audit CSV file
    
    Selects response pairs from the same group and saves them
    for human review of semantic consistency"""
    prompts = load_jsonl(dataset_path)
    
    # Load raw outputs if available
    raw_outputs = {}
    if Path(raw_output_path).exists():
        with open(raw_output_path) as f:
            for line in f:
                record = json.loads(line)
                raw_outputs[record["prompt_id"]] = record
    
    # Group prompts
    groups = defaultdict(list)
    for p in prompts:
        group_key = p.get("group_id", p["prompt_id"])
        groups[group_key].append(p)
    
    # Create audit pairs
    audit_pairs = []
    
    for group_id, group_prompts in sorted(groups.items()):
        if len(group_prompts) < 2:
            continue  # Skip single-prompt groups
        
        attack_type = group_prompts[0].get("attack_type", "unknown")
        
        for i in range(len(group_prompts)):
            for j in range(i + 1, len(group_prompts)):
                p1 = group_prompts[i]
                p2 = group_prompts[j]
                
                r1 = raw_outputs.get(p1["prompt_id"], {})
                r2 = raw_outputs.get(p2["prompt_id"], {})
                
                resp1 = r1.get("response", "")
                resp2 = r2.get("response", "")
                
                # Auto-classify
                actual1 = classify_response(resp1)
                actual2 = classify_response(resp2)
                
                if attack_type == "perturbation":
                    label_match = actual1 == "comply" and actual2 == "comply"
                else:
                    label_match = actual1 == actual2
                
                auto_label = "consistent" if label_match else "inconsistent"
                
                audit_pairs.append({
                    "group_id": group_id,
                    "attack_type": attack_type,
                    "prompt_text_1": p1["prompt_text"],
                    "response_1": resp1,
                    "prompt_text_2": p2["prompt_text"],
                    "response_2": resp2,
                    "auto_label": auto_label,
                    "human_label": ""  # To be filled manually
                })
    
    # Limit to 30 pairs
    audit_pairs = audit_pairs[:30]
    
    # Save as CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "group_id", "attack_type", "prompt_text_1", "response_1",
            "prompt_text_2", "response_2", "auto_label", "human_label"
        ])
        writer.writeheader()
        writer.writerows(audit_pairs)
    
    print(f"Saved {len(audit_pairs)} audit pairs to {output_path}")
    print(f"   Open this CSV file and fill in the 'human_label' column.")
    print(f"   Use 'consistent' or 'inconsistent' for each pair.")
    print()
    print("Preview:")
    for pair in audit_pairs[:5]:
        print(f"  {pair['group_id']} ({pair['attack_type']}): auto={pair['auto_label']}")


if __name__ == "__main__":
    create_audit_file()