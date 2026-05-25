"""Build a reference corpus file from a directory of .md and .txt files.

Usage:
    python build_corpus.py <input_dir> <output_file>

Scans the input directory recursively for .md and .txt files, splits them
into paragraphs (blank-line separated), filters by length, and writes one
paragraph per line to the output file.
"""
from __future__ import annotations

import sys
from pathlib import Path


def build_corpus(input_dir: Path, output_file: Path,
                 min_len: int = 60, max_len: int = 500) -> int:
    paragraphs: list[str] = []
    for p in input_dir.rglob("*"):
        if p.suffix.lower() not in {".md", ".txt"}:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except Exception:
            continue
        for chunk in text.split("\n\n"):
            chunk = " ".join(chunk.split())  # collapse whitespace
            if min_len <= len(chunk) <= max_len and "```" not in chunk:
                paragraphs.append(chunk)

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for p in paragraphs:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    output_file.write_text("\n".join(unique), encoding="utf-8")
    return len(unique)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} <input_dir> <output_file>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_file = Path(sys.argv[2])

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        sys.exit(1)

    count = build_corpus(input_dir, output_file)
    print(f"Wrote {count} unique paragraphs to {output_file}")
    if count < 200:
        print(f"WARNING: Only {count} paragraphs. Need at least 200, recommend 2000+.")
    elif count < 1200:
        print(f"NOTE: {count} paragraphs is workable but below the 1200 recommended minimum for stable 384-dim whitening.")
