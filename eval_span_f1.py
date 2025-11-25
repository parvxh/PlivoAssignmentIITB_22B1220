"""
Span-level F1 evaluation for PII NER.
"""
import json
import argparse
from collections import defaultdict
from typing import List, Dict, Set, Tuple


def load_jsonl_gold(path: str) -> Dict[str, List[Dict]]:
    """Load gold annotations from JSONL."""
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            uid = obj["id"]
            entities = obj.get("entities", [])
            data[uid] = entities
    return data


def load_json_pred(path: str) -> Dict[str, List[Dict]]:
    """Load predictions from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def span_to_tuple(entity: Dict) -> Tuple[int, int, str]:
    """Convert entity dict to comparable tuple."""
    return (entity["start"], entity["end"], entity["label"])


def compute_metrics(gold_spans: Set[Tuple], pred_spans: Set[Tuple]) -> Dict[str, float]:
    """Compute precision, recall, F1 for a set of spans."""
    if len(pred_spans) == 0:
        precision = 0.0
    else:
        precision = len(gold_spans & pred_spans) / len(pred_spans)
    
    if len(gold_spans) == 0:
        recall = 0.0
    else:
        recall = len(gold_spans & pred_spans) / len(gold_spans)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": len(gold_spans),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate span-level F1")
    parser.add_argument("--gold", required=True, help="Gold JSONL file")
    parser.add_argument("--pred", required=True, help="Predictions JSON file")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("📊 Span-Level F1 Evaluation")
    print("=" * 60)
    print(f"Gold: {args.gold}")
    print(f"Pred: {args.pred}")
    print("=" * 60 + "\n")
    
    # Load data
    gold_data = load_jsonl_gold(args.gold)
    pred_data = load_json_pred(args.pred)
    
    # Collect spans by label
    all_gold_spans = set()
    all_pred_spans = set()
    
    gold_by_label = defaultdict(set)
    pred_by_label = defaultdict(set)
    
    pii_gold_spans = set()
    pii_pred_spans = set()
    
    PII_LABELS = {"PHONE", "CREDIT_CARD", "EMAIL", "PERSON_NAME", "DATE"}
    
    for uid in gold_data:
        gold_entities = gold_data[uid]
        pred_entities = pred_data.get(uid, [])
        
        for entity in gold_entities:
            span = span_to_tuple(entity)
            all_gold_spans.add(span)
            gold_by_label[entity["label"]].add(span)
            
            if entity["label"] in PII_LABELS:
                pii_gold_spans.add(span)
        
        for entity in pred_entities:
            span = span_to_tuple(entity)
            all_pred_spans.add(span)
            pred_by_label[entity["label"]].add(span)
            
            if entity["label"] in PII_LABELS:
                pii_pred_spans.add(span)
    
    # Compute overall metrics
    overall_metrics = compute_metrics(all_gold_spans, all_pred_spans)
    
    print("🌍 Overall Metrics:")
    print(f"   Precision: {overall_metrics['precision']:.4f}")
    print(f"   Recall:    {overall_metrics['recall']:.4f}")
    print(f"   F1:        {overall_metrics['f1']:.4f}")
    print(f"   Support:   {overall_metrics['support']}")
    print()
    
    # Compute PII metrics
    pii_metrics = compute_metrics(pii_gold_spans, pii_pred_spans)
    
    print("🔒 PII Metrics (PHONE, CREDIT_CARD, EMAIL, PERSON_NAME, DATE):")
    print(f"   Precision: {pii_metrics['precision']:.4f}")
    print(f"   Recall:    {pii_metrics['recall']:.4f}")
    print(f"   F1:        {pii_metrics['f1']:.4f}")
    print(f"   Support:   {pii_metrics['support']}")
    print()
    
    # Per-label metrics
    print("📋 Per-Label Metrics:")
    print("-" * 60)
    
    all_labels = sorted(set(gold_by_label.keys()) | set(pred_by_label.keys()))
    
    for label in all_labels:
        gold_spans = gold_by_label[label]
        pred_spans = pred_by_label[label]
        
        metrics = compute_metrics(gold_spans, pred_spans)
        
        pii_marker = "🔒" if label in PII_LABELS else "  "
        
        print(f"{pii_marker} {label:15s} | P: {metrics['precision']:.3f} | "
              f"R: {metrics['recall']:.3f} | F1: {metrics['f1']:.3f} | "
              f"Support: {metrics['support']:3d}")
    
    print("-" * 60)
    print()
    
    # Summary
    print("=" * 60)
    print("✅ Evaluation Complete")
    print("=" * 60)
    print(f"Overall F1:     {overall_metrics['f1']:.4f}")
    print(f"PII F1:         {pii_metrics['f1']:.4f}")
    print(f"PII Precision:  {pii_metrics['precision']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()