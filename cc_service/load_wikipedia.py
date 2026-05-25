"""Load Simple English Wikipedia into the memory bank.

Downloads via HuggingFace `datasets`, splits articles into paragraphs,
and batch-posts them to the running service.

Usage:
    python load_wikipedia.py [--max-articles N] [--min-chars 80] [--batch-size 50]
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
MIN_CHARS = 80


def split_paragraphs(text: str, min_chars: int) -> list[str]:
    """Split article text into paragraphs, keeping only substantial ones."""
    chunks = []
    for para in text.split("\n\n"):
        clean = " ".join(para.strip().split())
        if len(clean) >= min_chars:
            chunks.append(clean)
    return chunks


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-articles", type=int, default=None,
                        help="Limit number of articles (default: all)")
    parser.add_argument("--skip-articles", type=int, default=0,
                        help="Skip this many articles (to resume a partial load)")
    parser.add_argument("--min-chars", type=int, default=MIN_CHARS,
                        help="Minimum paragraph length")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Cells per API call")
    parser.add_argument("--base-url", type=str, default=BASE_URL)
    args = parser.parse_args()

    # Check service
    client = httpx.Client(base_url=args.base_url, timeout=600.0)
    try:
        info = client.get("/info").json()
        if not info.get("initialized"):
            console.print("[red]Bank not initialized. Run 'ccmem init' first.[/red]")
            sys.exit(1)
        console.print(f"Bank has [cyan]{info['n_cells']}[/cyan] cells before loading.")
    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to {args.base_url}. Is the service running?[/red]")
        sys.exit(1)

    # Load dataset
    console.print("[cyan]Downloading Simple English Wikipedia...[/cyan]")
    console.print("(First run downloads ~180MB; cached after that)")
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")

    total_articles = len(ds)
    if args.max_articles:
        total_articles = min(total_articles, args.skip_articles + args.max_articles)
    console.print(f"Processing [cyan]{total_articles - args.skip_articles}[/cyan] articles (skipping first {args.skip_articles})...")

    # Extract and post
    loaded = 0
    errors = 0
    skipped = 0
    batch_texts: list[str] = []
    batch_labels: list[str | None] = []
    start = time.time()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Loading articles...", total=total_articles - args.skip_articles)

        for i, article in enumerate(ds):
            if i < args.skip_articles:
                continue
            if i >= total_articles:
                break

            title = article.get("title", "")
            text = article.get("text", "")
            paragraphs = split_paragraphs(text, args.min_chars)

            if not paragraphs:
                skipped += 1
                progress.advance(task)
                continue

            for para in paragraphs:
                batch_texts.append(para)
                batch_labels.append(title)

                if len(batch_texts) >= args.batch_size:
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
                                console.print(f"\n[red]Batch error (article ~{i}): {e}[/red]")
                            else:
                                time.sleep(1)
                    batch_texts.clear()
                    batch_labels.clear()

            progress.advance(task)
            progress.update(task,
                description=f"Loading... {loaded} cells")

        # Flush remaining batch
        if batch_texts:
            try:
                resp = client.post("/write_many", json={
                    "texts": batch_texts,
                    "labels": batch_labels,
                })
                resp.raise_for_status()
                data = resp.json()
                loaded += len(data["cell_ids"])
            except Exception as e:
                errors += len(batch_texts)
                console.print(f"\n[red]Final batch error: {e}[/red]")

    elapsed = time.time() - start
    console.print(f"\n[green]✓ Loaded {loaded:,} cells in {elapsed/60:.1f} minutes[/green]")
    if skipped:
        console.print(f"  Skipped {skipped:,} articles (no paragraphs above {args.min_chars} chars)")
    if errors:
        console.print(f"  [red]{errors:,} errors[/red]")

    # Final count
    try:
        info = client.get("/info").json()
        console.print(f"  Bank now has [cyan]{info['n_cells']:,}[/cyan] cells total.")
    except Exception:
        pass


if __name__ == "__main__":
    main()
