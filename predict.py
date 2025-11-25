import json
import argparse
import torch
import os
from transformers import AutoTokenizer
from labels import ID2LABEL
from model import create_model


def bio_to_spans(offsets, preds):
    """
    Convert BIO predictions into entity spans.
    offsets: [(start_char, end_char), ...]
    preds: [label_id, ...]
    Returns: [(start, end, label), ...]
    """
    spans = []
    cur_label = None
    cur_start = None
    cur_end = None

    for (start, end), p in zip(offsets, preds):

        # Skip special tokens ([CLS], [SEP])
        if start == 0 and end == 0:
            continue

        label = ID2LABEL.get(p, "O")

        if label == "O":
            if cur_label is not None:
                spans.append((cur_start, cur_end, cur_label))
                cur_label = None
            continue

        prefix, ent = label.split("-", 1)

        if prefix == "B":
            # If we were inside an entity → close it
            if cur_label is not None:
                spans.append((cur_start, cur_end, cur_label))
            # Start new one
            cur_label = ent
            cur_start = start
            cur_end = end

        elif prefix == "I":
            if cur_label == ent:
                # Continue same entity
                cur_end = end
            else:
                # Broken I- (start new)
                if cur_label is not None:
                    spans.append((cur_start, cur_end, cur_label))
                cur_label = ent
                cur_start = start
                cur_end = end

    # Add final span
    if cur_label is not None:
        spans.append((cur_start, cur_end, cur_label))

    return spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="out")
    parser.add_argument("--input", default="data/dev.jsonl")
    parser.add_argument("--output", default="out/dev_pred.json")
    parser.add_argument("--max_length", type=int, default=256)
    args = parser.parse_args()

    print("🔍 Loading tokenizer + model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = create_model(args.model_dir, num_labels=len(ID2LABEL))
    model.to("cpu")
    model.eval()

    results = {}

    print("📝 Reading input:", args.input)
    with open(args.input) as f:
        lines = [json.loads(x) for x in f]

    for ex in lines:
        text = ex["text"]

        enc = tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

        offsets = enc["offset_mapping"][0].tolist()

        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            ).logits[0]

        preds = logits.argmax(-1).tolist()

        spans = bio_to_spans(offsets, preds)

        ents = []
        for s, e, lab in spans:
            ents.append({
                "start": int(s),
                "end": int(e),
                "label": lab,
                "pii": lab != "O"
            })

        results[ex["id"]] = ents

    # Save output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print("✅ Saved predictions to:", args.output)


if __name__ == "__main__":
    main()
