import json
from torch.utils.data import Dataset

class PIIDataset(Dataset):
    """
    Correct dataset:
    - Assign BIO tags based on token/char overlap
    - Works with any HF token classifier
    """

    def __init__(self, path, tokenizer, label_list, max_length=256):
        self.items = []
        self.label2id = {l: i for i,l in enumerate(label_list)}
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(path, "r") as f:
            for line in f:
                ex = json.loads(line)
                text = ex["text"]
                spans = ex["entities"]

                # tokenize
                enc = tokenizer(
                    text,
                    return_offsets_mapping=True,
                    truncation=True,
                    max_length=max_length,
                    add_special_tokens=True,
                )
                offsets = enc["offset_mapping"]

                labels = []
                for (s, e) in offsets:
                    if s == 0 and e == 0:
                        labels.append(self.label2id["O"])
                        continue

                    assigned = "O"
                    for sp in spans:
                        es, ee, lab = sp["start"], sp["end"], sp["label"]
                        if not (e <= es or s >= ee):  # overlap
                            assigned = "B-" + lab if assigned == "O" else "I-" + lab
                    labels.append(self.label2id.get(assigned, 0))

                self.items.append({
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "labels": labels,
                    "id": ex["id"],
                    "text": text,
                    "offsets": offsets,
                })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_batch(batch, pad_token_id, label_pad_id=-100):
    max_len = max(len(x["input_ids"]) for x in batch)

    def pad(seq, val):
        return seq + [val] * (max_len - len(seq))

    return {
        "input_ids": [pad(x["input_ids"], pad_token_id) for x in batch],
        "attention_mask": [pad(x["attention_mask"], 0) for x in batch],
        "labels": [pad(x["labels"], label_pad_id) for x in batch],
        "ids": [x["id"] for x in batch],
        "texts": [x["text"] for x in batch],
        "offsets": [x["offsets"] for x in batch],
    }
