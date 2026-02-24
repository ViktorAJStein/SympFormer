
import argparse
import os
from array import array

import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--max_docs_train", type=int, default=0, help="0 = all")
    ap.add_argument("--max_docs_val", type=int, default=0, help="0 = all")
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

    ds = load_dataset("roneneldan/TinyStories")
    train = ds["train"]
    val = ds["validation"]

    def dump(split, dset, max_docs, out_path):
        n = len(dset) if max_docs == 0 else min(len(dset), max_docs)
        print(f"[{split}] docs={n} -> {out_path}")
        with open(out_path, "wb") as f:
            buf = array("H")
            for i in range(n):
                text = dset[i]["text"]
                toks = enc.encode_ordinary(text)
                toks.append(eot)
                # ensure uint16
                if max(toks) >= 65535:
                    raise ValueError("Token id exceeds uint16 range")
                buf.extend(toks)
                # flush periodically
                if len(buf) > 1_000_000:
                    buf.tofile(f)
                    buf = array("H")
                if (i + 1) % 1000 == 0:
                    print(f"  processed {i+1}/{n}")
            if len(buf) > 0:
                buf.tofile(f)
        # report token count
        arr = np.memmap(out_path, dtype=np.uint16, mode="r")
        print(f"[{split}] tokens={len(arr)}")

    dump("train", train, args.max_docs_train, os.path.join(args.out_dir, "tinystories_train.bin"))
    dump("val", val, args.max_docs_val, os.path.join(args.out_dir, "tinystories_val.bin"))


if __name__ == "__main__":
    main()
