"""Load PDFs into the concept-cells memory bank.

Extracts text from one or more PDFs, splits into paragraphs, filters for
quality, and loads them via the service API.

Usage:
    # Single PDF
    python load_pdf.py paper.pdf

    # A folder of PDFs
    python load_pdf.py ~/Papers/

    # With a label prefix (helps filter later)
    python load_pdf.py ~/Papers/ --label-prefix "papers"

    # Dry run — see what would be loaded without loading it
    python load_pdf.py paper.pdf --dry-run

    # Adjust paragraph filtering
    python load_pdf.py paper.pdf --min-chars 80 --max-chars 800

Requirements:
    pip install pypdf httpx typer rich
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import httpx
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


# ---------- PDF text extraction ----------

def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using pypdf.

    Returns the full text as a single string. Handles common issues:
    - Gracefully skips pages that fail extraction
    - Strips null bytes and control characters
    """
    from pypdf import PdfReader

    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        console.print(f"[red]Cannot open {pdf_path.name}: {e}[/red]")
        return ""

    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
            if text:
                pages_text.append(text)
        except Exception as e:
            console.print(
                f"[yellow]  skipping page {i+1} of {pdf_path.name}: {e}[/yellow]"
            )
    return "\n\n".join(pages_text)


# ---------- Paragraph splitting and filtering ----------

# Patterns that indicate junk rather than real content
_JUNK_PATTERNS = [
    re.compile(r"^\s*\d+\s*$"),                           # bare page numbers
    re.compile(r"^(figure|table|fig\.|tab\.)\s+\d", re.I), # figure/table captions (short)
    re.compile(r"^\s*references\s*$", re.I),               # section headers
    re.compile(r"^\s*abstract\s*$", re.I),
    re.compile(r"^\s*acknowledgements?\s*$", re.I),
    re.compile(r"^\s*https?://"),                           # bare URLs
    re.compile(r"^[•●■▪►▸\-–—]\s"),                        # bullet list fragments
]


def split_into_paragraphs(text: str, min_chars: int = 80,
                           max_chars: int = 800) -> List[str]:
    """Split extracted PDF text into usable paragraphs.

    PDF text extraction often produces messy output: hyphenated line breaks,
    headers mixed with body text, reference lists, etc. We do our best to
    clean it up and filter for paragraphs that are likely to be meaningful
    content.
    """
    # Normalise whitespace within lines (PDF extractors leave weird spaces)
    text = re.sub(r"[ \t]+", " ", text)

    # Rejoin hyphenated line breaks: "some-\nword" → "someword"
    text = re.sub(r"-\n(\S)", r"\1", text)

    # Split on blank lines (the primary paragraph delimiter)
    raw_chunks = re.split(r"\n\s*\n", text)

    paragraphs = []
    for chunk in raw_chunks:
        # Collapse remaining newlines into spaces (within a paragraph)
        chunk = chunk.replace("\n", " ").strip()

        # Length filter
        if len(chunk) < min_chars:
            continue
        if len(chunk) > max_chars:
            # For very long chunks, try to split on sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", chunk)
            current = ""
            for sent in sentences:
                if len(current) + len(sent) + 1 > max_chars and len(current) >= min_chars:
                    paragraphs.append(current.strip())
                    current = sent
                else:
                    current = (current + " " + sent).strip()
            if len(current) >= min_chars:
                paragraphs.append(current.strip())
            continue

        # Junk filter
        if any(pat.match(chunk) for pat in _JUNK_PATTERNS):
            continue

        # References section heuristic: lines that look like bibliography entries
        # (start with "[N]" or "N." followed by author names)
        if re.match(r"^\[?\d{1,3}\]?\s*[A-Z][a-z]+", chunk) and "doi" in chunk.lower():
            continue

        paragraphs.append(chunk)

    return paragraphs


def process_one_pdf(pdf_path: Path, min_chars: int, max_chars: int,
                     label_prefix: Optional[str]) -> List[Tuple[str, str]]:
    """Extract and split one PDF. Returns list of (text, label) pairs."""
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        console.print(f"[yellow]  {pdf_path.name}: no extractable text (scanned?)[/yellow]")
        return []

    paragraphs = split_into_paragraphs(text, min_chars=min_chars,
                                         max_chars=max_chars)

    # Build the label: [prefix] filename
    stem = pdf_path.stem[:60]  # cap filename length in label
    label = f"[{label_prefix}] {stem}" if label_prefix else stem

    return [(p, label) for p in paragraphs]


# ---------- API interaction ----------

def load_to_bank(items: List[Tuple[str, str]], base_url: str,
                  batch_size: int = 50) -> int:
    """Send paragraphs to the bank via /write_many in batches."""
    loaded = 0
    with httpx.Client(base_url=base_url, timeout=120.0) as client:
        # Check health
        try:
            r = client.get("/health")
            r.raise_for_status()
        except Exception as e:
            console.print(f"[red]Service unreachable at {base_url}: {e}[/red]")
            raise typer.Exit(code=1)

        # Check initialized
        info = client.get("/info").json()
        if not info.get("initialized"):
            console.print("[red]Bank not initialized. Run `ccmem init` first.[/red]")
            raise typer.Exit(code=1)

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading to bank...", total=len(items))

            for i in range(0, len(items), batch_size):
                batch = items[i : i + batch_size]
                texts = [t for t, _ in batch]
                labels = [l for _, l in batch]
                try:
                    r = client.post("/write_many",
                                      json={"texts": texts, "labels": labels})
                    r.raise_for_status()
                    loaded += len(r.json()["cell_ids"])
                except Exception as e:
                    console.print(f"\n[red]Error on batch {i//batch_size}: {e}[/red]")
                    # Continue with next batch rather than aborting
                progress.advance(task, len(batch))

    return loaded


# ---------- CLI ----------

@app.command()
def load(
    path: Path = typer.Argument(
        ..., help="Path to a PDF file or a directory containing PDFs.",
        exists=True,
    ),
    label_prefix: Optional[str] = typer.Option(
        None, "--label-prefix", "-l",
        help="Prefix for cell labels, e.g. 'papers' → '[papers] filename'",
    ),
    min_chars: int = typer.Option(
        80, "--min-chars",
        help="Minimum paragraph length to include.",
    ),
    max_chars: int = typer.Option(
        800, "--max-chars",
        help="Maximum paragraph length (longer ones get sentence-split).",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Extract and show stats but don't load to the bank.",
    ),
    base_url: str = typer.Option(
        "http://127.0.0.1:8765", "--url",
        help="Base URL of the ccmem service.",
    ),
    batch_size: int = typer.Option(
        50, "--batch-size",
        help="Number of paragraphs per API call.",
    ),
):
    """Extract text from PDFs and load into the concept-cells bank.

    Accepts a single PDF file or a directory (searches recursively for *.pdf).
    """
    # Gather PDF files
    if path.is_file():
        if path.suffix.lower() != ".pdf":
            console.print(f"[red]{path} is not a PDF file.[/red]")
            raise typer.Exit(code=1)
        pdf_files = [path]
    else:
        pdf_files = sorted(path.rglob("*.pdf"))
        if not pdf_files:
            console.print(f"[yellow]No PDFs found in {path}[/yellow]")
            raise typer.Exit(code=1)

    console.print(f"Found [cyan]{len(pdf_files)}[/cyan] PDF(s) in {path}\n")

    # Process each PDF
    all_items: List[Tuple[str, str]] = []
    for pdf_path in pdf_files:
        console.print(f"  Processing [bold]{pdf_path.name}[/bold]...")
        items = process_one_pdf(pdf_path, min_chars=min_chars,
                                  max_chars=max_chars,
                                  label_prefix=label_prefix)
        console.print(f"    → {len(items)} paragraphs")
        all_items.extend(items)

    # De-duplicate
    seen = set()
    unique_items = []
    for text, label in all_items:
        if text not in seen:
            seen.add(text)
            unique_items.append((text, label))
    n_dupes = len(all_items) - len(unique_items)
    if n_dupes > 0:
        console.print(f"\n  Removed {n_dupes} duplicate paragraphs.")

    console.print(f"\n[bold]Total: {len(unique_items)} unique paragraphs "
                    f"from {len(pdf_files)} PDF(s)[/bold]")

    if not unique_items:
        console.print("[yellow]Nothing to load.[/yellow]")
        return

    # Show a sample
    console.print("\n[dim]Sample paragraphs:[/dim]")
    for text, label in unique_items[:3]:
        console.print(f"  [{label}] {text[:120]}...")
    if len(unique_items) > 3:
        console.print(f"  [dim]... and {len(unique_items) - 3} more[/dim]")

    if dry_run:
        console.print("\n[yellow]Dry run — nothing loaded.[/yellow]")
        return

    # Load
    console.print()
    loaded = load_to_bank(unique_items, base_url=base_url,
                            batch_size=batch_size)
    console.print(f"\n[green]Done. Loaded {loaded} cells to the bank.[/green]")


if __name__ == "__main__":
    app()
