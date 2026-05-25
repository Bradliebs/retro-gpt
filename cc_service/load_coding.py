"""Load coding content into the memory bank.

Sources:
  - CodeAlpaca-20k: 20k programming instruction/response pairs

Usage:
    python load_coding.py [--max-items N]
"""
from __future__ import annotations

import sys
import time

import httpx
from datasets import load_dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

console = Console()
BASE_URL = "http://127.0.0.1:8765"
BATCH_SIZE = 50


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int, default=None)
    parser.add_argument("--base-url", type=str, default=BASE_URL)
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=600.0)
    try:
        info = client.get("/info").json()
        if not info.get("initialized"):
            console.print("[red]Bank not initialized.[/red]")
            sys.exit(1)
        console.print(f"Bank has [cyan]{info['n_cells']:,}[/cyan] cells before loading.")
    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to {args.base_url}[/red]")
        sys.exit(1)

    # Load CodeAlpaca-20k
    console.print("[cyan]Downloading CodeAlpaca-20k...[/cyan]")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

    total = len(ds)
    if args.max_items:
        total = min(total, args.max_items)
    console.print(f"Processing [cyan]{total:,}[/cyan] coding items...")

    loaded = 0
    errors = 0
    skipped = 0
    batch_texts: list[str] = []
    batch_labels: list[str | None] = []
    start = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading coding content...", total=total)

        for i, item in enumerate(ds):
            if i >= total:
                break

            instruction = (item.get("instruction") or "").strip()
            output = (item.get("output") or "").strip()

            # Skip items with very short or empty responses
            if len(output) < 40:
                skipped += 1
                progress.advance(task)
                continue

            # Use instruction as label, output as the cell text
            # For items with code, prepend the instruction for context
            if instruction:
                text = f"{instruction}\n\n{output}"
                label = f"[code] {instruction[:80]}"
            else:
                text = output
                label = "[code]"

            # Skip very long items (likely just giant code dumps)
            if len(text) > 3000:
                text = text[:3000]

            batch_texts.append(text)
            batch_labels.append(label)

            if len(batch_texts) >= BATCH_SIZE:
                for attempt in range(3):
                    try:
                        resp = client.post("/write_many", json={
                            "texts": batch_texts,
                            "labels": batch_labels,
                        })
                        resp.raise_for_status()
                        data = resp.json()
                        loaded += len(data["cell_ids"])
                        break
                    except Exception as e:
                        if attempt == 2:
                            errors += len(batch_texts)
                            console.print(f"\n[red]Batch error: {e}[/red]")
                        else:
                            time.sleep(1)
                batch_texts.clear()
                batch_labels.clear()

            progress.advance(task)
            progress.update(task, description=f"Loading... {loaded} cells")

        # Flush remaining
        if batch_texts:
            try:
                resp = client.post("/write_many", json={
                    "texts": batch_texts, "labels": batch_labels,
                })
                resp.raise_for_status()
                loaded += len(resp.json()["cell_ids"])
            except Exception as e:
                errors += len(batch_texts)
                console.print(f"\n[red]Final batch error: {e}[/red]")

    elapsed = time.time() - start
    console.print(f"\n[green]✓ Loaded {loaded:,} coding cells in {elapsed:.1f}s[/green]")
    if skipped:
        console.print(f"  Skipped {skipped:,} items (too short)")
    if errors:
        console.print(f"  [red]{errors:,} errors[/red]")

    try:
        info = client.get("/info").json()
        console.print(f"  Bank now has [cyan]{info['n_cells']:,}[/cyan] cells total.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
