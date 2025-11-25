import os, json, argparse, torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_scheduler
from torch.optim import AdamW

from dataset import PIIDataset, collate_batch
from labels import LABELS
from model import create_model
from tqdm import tqdm

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--dev", default="data/dev.jsonl")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    return ap.parse_args()

def main():
    args = parse()
    os.makedirs(args.out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_name)
    train_ds = PIIDataset(args.train, tok, LABELS)
    train_dl = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, tok.pad_token_id)
    )

    model = create_model(args.model_name, num_labels=len(LABELS))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    opt = AdamW(model.parameters(), lr=args.lr)
    num_steps = len(train_dl) * args.epochs
    scheduler = get_scheduler("linear", opt, 0, num_steps)

    for ep in range(args.epochs):
        total = 0
        for batch in tqdm(train_dl, desc=f"Epoch {ep+1}"):
            ids = torch.tensor(batch["input_ids"]).to(device)
            mask = torch.tensor(batch["attention_mask"]).to(device)
            labels = torch.tensor(batch["labels"]).to(device)

            opt.zero_grad()
            out = model(input_ids=ids, attention_mask=mask, labels=labels)
            loss = out.loss
            loss.backward()
            opt.step()
            scheduler.step()
            total += loss.item()

        print(f"Epoch {ep+1} avg loss = {total/len(train_dl)}")

    model.save_pretrained(args.out_dir)
    tok.save_pretrained(args.out_dir)
    print("Model saved.")

if __name__=="__main__":
    main()
