"""Bulk-load a folder of text/markdown files into the memory bank.

Usage:
    python -m cli.bulk_load <folder> [--ext .md .txt] [--min-chars 80] [--label-from filename]

Reads every file matching the extensions, splits into paragraphs,
filters by minimum length, and POSTs them to the running service.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import httpx
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

app = typer.Typer(add_completion=False)
console = Console()

BASE_URL = "http://127.0.0.1:8765"
BATCH_SIZE = 50


def _split_paragraphs(text: str, min_chars: int) -> List[str]:
    """Split text into paragraphs, keeping only those above min_chars."""
    chunks = []
    for para in text.split("\n\n"):
        clean = para.strip()
        if len(clean) >= min_chars:
            # Collapse internal whitespace
            clean = " ".join(clean.split())
            chunks.append(clean)
    return chunks


def _split_markdown_sections(text: str, min_chars: int) -> list[tuple[str | None, str]]:
    """Split markdown by headings, returning (heading, body) pairs."""
    results = []
    current_heading = None
    current_lines: list[str] = []

    for line in text.split("\n"):
        if line.startswith("#"):
            # Flush previous section
            body = "\n\n".join(p.strip() for p in "\n".join(current_lines).split("\n\n") if p.strip())
            if body and len(body) >= min_chars:
                results.append((current_heading, body))
            current_heading = line.lstrip("#").strip()
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    body = "\n\n".join(p.strip() for p in "\n".join(current_lines).split("\n\n") if p.strip())
    if body and len(body) >= min_chars:
        results.append((current_heading, body))

    return results


@app.command()
def load(
    folder: Path = typer.Argument(..., help="Folder containing text/markdown files"),
    ext: List[str] = typer.Option([".md", ".txt"], help="File extensions to include"),
    min_chars: int = typer.Option(80, help="Minimum paragraph length to keep"),
    label_from: str = typer.Option("filename", help="'filename' or 'heading' or 'none'"),
    by_section: bool = typer.Option(False, help="Split markdown by heading sections instead of paragraphs"),
    dry_run: bool = typer.Option(False, help="Show what would be loaded without sending"),
    base_url: str = typer.Option(BASE_URL, help="Service URL"),
):
    """Bulk-load a folder of documents into the memory bank."""
    if not folder.exists():
        console.print(f"[red]Folder not found: {folder}[/red]")
        raise typer.Exit(1)

    # Collect all files
    files = []
    for e in ext:
        files.extend(sorted(folder.rglob(f"*{e}")))

    if not files:
        console.print(f"[yellow]No files found with extensions {ext} in {folder}[/yellow]")
        raise typer.Exit(1)

    console.print(f"Found [cyan]{len(files)}[/cyan] files in [cyan]{folder}[/cyan]")

    # Extract paragraphs
    items: list[tuple[str, Optional[str]]] = []  # (text, label)
    for f in files:
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            console.print(f"[yellow]Skipping {f.name}: {e}[/yellow]")
            continue

        file_label = f.stem.replace("_", " ").replace("-", " ")

        if by_section and f.suffix == ".md":
            sections = _split_markdown_sections(content, min_chars)
            for heading, body in sections:
                if label_from == "heading" and heading:
                    items.append((body, heading))
                elif label_from == "filename":
                    label = f"{file_label} — {heading}" if heading else file_label
                    items.append((body, label))
                else:
                    items.append((body, None))
        else:
            paras = _split_paragraphs(content, min_chars)
            for p in paras:
                if label_from == "filename":
                    items.append((p, file_label))
                else:
                    items.append((p, None))

    if not items:
        console.print("[yellow]No paragraphs met the minimum length threshold.[/yellow]")
        raise typer.Exit(1)

    console.print(f"Extracted [cyan]{len(items)}[/cyan] chunks (min {min_chars} chars)")

    if dry_run:
        console.print("\n[yellow]DRY RUN — showing first 10 chunks:[/yellow]")
        for text, label in items[:10]:
            lbl = f" [{label}]" if label else ""
            console.print(f"  {text[:100]}...{lbl}")
        console.print(f"\n[yellow]Would load {len(items)} chunks total.[/yellow]")
        return

    # Check service health
    client = httpx.Client(base_url=base_url, timeout=60.0)
    try:
        info = client.get("/info").json()
        if not info.get("initialized"):
            console.print("[red]Bank not initialized. Run 'ccmem init' first.[/red]")
            raise typer.Exit(1)
    except httpx.ConnectError:
        console.print(f"[red]Cannot connect to {base_url}. Is the service running?[/red]")
        raise typer.Exit(1)

    # Send in batches
    loaded = 0
    errors = 0
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(), TaskProgressColumn(), console=console,
    ) as progress:
        task = progress.add_task("Loading cells...", total=len(items))
        for i in range(0, len(items), BATCH_SIZE):
            batch = items[i : i + BATCH_SIZE]
            texts = [t for t, _ in batch]
            labels = [l for _, l in batch]
            try:
                resp = client.post(
                    "/write_many",
                    json={"texts": texts, "labels": labels},
                    timeout=120.0,
                )
                resp.raise_for_status()
                data = resp.json()
                loaded += len(data["cell_ids"])
            except Exception as e:
                errors += len(batch)
                console.print(f"[red]Batch error: {e}[/red]")
            progress.advance(task, len(batch))

    console.print(f"\n[green]✓ Loaded {loaded} cells[/green]", end="")
    if errors:
        console.print(f" [red]({errors} errors)[/red]")
    else:
        console.print()

    # Show final count
    try:
        info = client.get("/info").json()
        console.print(f"Bank now has [cyan]{info['n_cells']}[/cyan] cells total.")
    except Exception:
        pass


if __name__ == "__main__":
    app()
