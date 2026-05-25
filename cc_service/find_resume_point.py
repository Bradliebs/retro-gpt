"""Find the right --skip-articles value to resume the Wikipedia loader.

Reads the most recently written cells from the bank, looks up their article
titles in the Simple Wikipedia dataset, and prints the highest matching index.
"""
from __future__ import annotations

import sys

import httpx
from datasets import load_dataset
from rich.console import Console

console = Console()
BASE_URL = "http://127.0.0.1:8765"
TAIL_CELLS = 50  # how many of the most recent cells to sample


def main():
    client = httpx.Client(base_url=BASE_URL, timeout=60.0)
    try:
        info = client.get("/info").json()
    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to {BASE_URL}.[/red]")
        sys.exit(1)

    total = info["n_cells"]
    console.print(f"Bank has [cyan]{total:,}[/cyan] cells.")

    # Fetch the tail of the cell list (highest offset = most recent ids).
    offset = max(0, total - TAIL_CELLS)
    resp = client.get("/cells", params={"limit": TAIL_CELLS, "offset": offset})
    resp.raise_for_status()
    cells = resp.json()["cells"]

    recent_labels = [c["label"] for c in cells if c.get("label")]
    if not recent_labels:
        console.print("[red]No labels found on recent cells.[/red]")
        sys.exit(1)

    console.print(f"Sampled [cyan]{len(recent_labels)}[/cyan] recent labels. Last 5:")
    for lab in recent_labels[-5:]:
        console.print(f"  - {lab}")

    console.print("\n[cyan]Loading Simple English Wikipedia (cached)...[/cyan]")
    ds = load_dataset("wikimedia/wikipedia", "20231101.simple", split="train")
    console.print(f"Dataset has [cyan]{len(ds):,}[/cyan] articles.")

    unique_labels = set(recent_labels)
    title_to_idx: dict[str, int] = {}

    console.print("[cyan]Scanning dataset for matching titles...[/cyan]")
    for i, art in enumerate(ds):
        title = art.get("title", "")
        if title in unique_labels:
            title_to_idx[title] = i
            if len(title_to_idx) == len(unique_labels):
                break

    if not title_to_idx:
        console.print("[red]No recent labels matched any dataset title.[/red]")
        console.print("[yellow]Falling back to estimate from cell count.[/yellow]")
        avg_paras = total / 112000 if total else 1
        est = int(total / max(avg_paras, 1))
        console.print(f"Rough estimate: --skip-articles {est}")
        sys.exit(0)

    max_idx = max(title_to_idx.values())
    found = len(title_to_idx)
    missing = len(unique_labels) - found

    console.print(
        f"\nMatched [cyan]{found}/{len(unique_labels)}[/cyan] titles "
        f"({missing} not found — likely paragraph titles from already-loaded articles)."
    )
    console.print(f"Highest matched index: [cyan]{max_idx:,}[/cyan]")

    # Resume one past the last seen article. Subtract a small safety margin
    # in case the very last article was only partially written before the crash.
    suggested = max(0, max_idx - 1)
    console.print(
        f"\n[green]Resume with:[/green] "
        f"--skip-articles {suggested}"
    )
    console.print(
        f"[dim](This will re-process the last 1-2 articles; duplicates become "
        f"extra cells but cause no harm.)[/dim]"
    )


if __name__ == "__main__":
    main()
