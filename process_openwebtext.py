import argparse
import hashlib
import json
import os
from array import array

import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--val_fraction", type=float, default=0.005, help="fraction of docs for validation (OpenWebText has only train split)")
    ap.add_argument("--max_docs_train", type=int, default=0, help="0 = no limit")
    ap.add_argument("--max_docs_val", type=int, default=0, help="0 = no limit")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    try:
        import tiktoken
    except ImportError as e:
        raise SystemExit("Please install tiktoken to run preprocessing.") from e

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise SystemExit("Please install datasets (huggingface) to run preprocessing.") from e

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256

    ds = load_dataset("Skylion007/openwebtext", split="train")  # single split
    if not 0.0 < args.val_fraction < 1.0:
        raise SystemExit("--val_fraction must lie in (0,1)")
    split = ds.train_test_split(
        test_size=args.val_fraction, seed=args.seed, shuffle=True
    )
    train = split["train"]
    val = split["test"]

    def dump(split, dset, max_docs, out_path):
        n = len(dset) if max_docs == 0 else min(len(dset), max_docs)
        print(f"[{split}] docs={n} -> {out_path}")
        digest = hashlib.sha256()
        with open(out_path, "wb") as f:
            buf = array("H")
            for i in range(n):
                text = dset[i]["text"]
                toks = enc.encode_ordinary(text)
                toks.append(eot)
                if toks and max(toks) >= 65535:
                    raise ValueError("Token id exceeds uint16 range")
                buf.extend(toks)
                if len(buf) > 1_000_000:
                    raw = buf.tobytes()
                    f.write(raw)
                    digest.update(raw)
                    buf = array("H")
                if (i + 1) % 5000 == 0:
                    print(f"  processed {i+1}/{n}")
            if len(buf) > 0:
                raw = buf.tobytes()
                f.write(raw)
                digest.update(raw)
        arr = np.memmap(out_path, dtype=np.uint16, mode="r")
        manifest = {
            "source": "Skylion007/openwebtext",
            "split": split,
            "documents": n,
            "tokens": len(arr),
            "bytes": os.path.getsize(out_path),
            "sha256": digest.hexdigest(),
            "dtype": "uint16",
            "tokenizer": "tiktoken:gpt2",
            "eot_token": eot,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
        }
        with open(out_path + ".manifest.json", "w", encoding="utf-8") as mf:
            json.dump(manifest, mf, indent=2, sort_keys=True)
        print(f"[{split}] tokens={len(arr)} sha256={digest.hexdigest()}")

    dump("train", train, args.max_docs_train, os.path.join(args.out_dir, "openwebtext_train.bin"))
    dump("val", val, args.max_docs_val, os.path.join(args.out_dir, "openwebtext_val.bin"))

if __name__ == "__main__":
    main()