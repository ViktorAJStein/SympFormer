#!/usr/bin/env python3
"""Render persisted sample JSONL records into auditable Markdown and LaTeX."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {"baseline": "Baseline", "yurii_lt": "YuriiFormer", "causal_symp_pe": "Causal PE SympFormer"}


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
        "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text).replace("\n", "\n\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument("--latex", type=Path, required=True)
    args = parser.parse_args()

    records = []
    for path in sorted(args.input_dir.glob("*.jsonl")):
        lines = [line for line in path.read_text().splitlines() if line.strip()]
        if len(lines) != 1:
            raise SystemExit(f"expected exactly one record in {path}, found {len(lines)}")
        records.append(json.loads(lines[0]))
    order = {"baseline": 0, "yurii_lt": 1, "causal_symp_pe": 2}
    records.sort(key=lambda record: order.get(record["arch"], 99))
    if len({tuple(record["prompt_token_ids"]) for record in records}) != 1:
        raise SystemExit("records do not share one prompt")

    prompt = records[0]["prompt_text"] or str(records[0]["prompt_token_ids"])
    md = ["# Deterministic generated examples", "", "## Common prompt", "", prompt, ""]
    tex = [r"\paragraph{Common prompt}", latex_escape(prompt), ""]
    for record in records:
        label = LABELS.get(record["arch"], record["arch"])
        continuation = record["continuation_text"] or str(record["continuation_token_ids"])
        md.extend([f"## {label}", "", continuation, ""])
        tex.extend([rf"\paragraph{{{latex_escape(label)}}}", latex_escape(continuation), ""])
    args.markdown.write_text("\n".join(md), encoding="utf-8")
    args.latex.write_text("\n".join(tex), encoding="utf-8")
    print(f"wrote {args.markdown} and {args.latex} ({len(records)} samples)")


if __name__ == "__main__":
    main()
