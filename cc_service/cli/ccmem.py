"""ccmem CLI — terminal client for the concept-cells memory service.

Usage:
    ccmem init --reference-corpus wikitext --reference-n 2000
    ccmem add "some text"
    ccmem add --file notes.txt
    ccmem query "what is X"
    ccmem bind 3 7 12 --label "topic name"
    ccmem list [--kind single|bound]
    ccmem show 7
    ccmem delete 7
    ccmem info
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    add_completion=False,
    help="Concept Cells memory CLI.",
    no_args_is_help=True,
)
console = Console()


DEFAULT_BASE_URL = os.environ.get("CCMEM_URL", "http://127.0.0.1:8765")


def _client(timeout: float = 60.0) -> httpx.Client:
    return httpx.Client(base_url=DEFAULT_BASE_URL, timeout=timeout)


def _handle_response(r: httpx.Response):
    """Pretty-print error responses, return JSON on success."""
    if r.is_success:
        return r.json()
    try:
        detail = r.json().get("detail", r.text)
    except Exception:
        detail = r.text
    console.print(f"[red]Error {r.status_code}:[/red] {detail}")
    raise typer.Exit(code=1)


# ---------- info & health ----------

@app.command()
def info():
    """Show service info: encoder, dim, n_cells, init state."""
    with _client() as c:
        data = _handle_response(c.get("/info"))
    console.print(Panel.fit(
        "\n".join(f"{k}: [cyan]{v}[/cyan]" for k, v in data.items()),
        title="ccmem info",
    ))


@app.command()
def health():
    """Check that the service is reachable."""
    try:
        with _client() as c:
            r = c.get("/health")
            r.raise_for_status()
        console.print("[green]Service is up.[/green]")
    except Exception as e:
        console.print(f"[red]Service unreachable: {e}[/red]")
        raise typer.Exit(code=1)


# ---------- init ----------

@app.command()
def init(
    reference_corpus: str = typer.Option(
        "wikitext", "--reference-corpus",
        help="Where to source whitening-reference texts: wikitext|file|stdin",
    ),
    reference_file: Optional[Path] = typer.Option(
        None, "--file",
        help="If --reference-corpus=file, path to a file with one text per line.",
    ),
    reference_n: int = typer.Option(
        2000, "--reference-n",
        help="Number of reference texts to load (min 200).",
    ),
):
    """Initialize the bank's whitening parameters from a reference corpus.

    This is a one-time setup. Once initialized, the bank's geometry is fixed
    and you can start writing notes.
    """
    if reference_n < 200:
        console.print("[red]reference_n must be at least 200.[/red]")
        raise typer.Exit(code=1)

    texts: List[str] = []
    if reference_corpus == "wikitext":
        console.print(f"Loading {reference_n} reference texts from wikitext...")
        try:
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                                split="train", streaming=True)
            for row in ds:
                t = row["text"].strip()
                if len(t) > 40 and not t.startswith("="):
                    texts.append(t[:512])
                    if len(texts) >= reference_n:
                        break
        except Exception as e:
            console.print(f"[red]Could not load wikitext: {e}[/red]")
            raise typer.Exit(code=1)
    elif reference_corpus == "file":
        if reference_file is None:
            console.print("[red]--file required when --reference-corpus=file.[/red]")
            raise typer.Exit(code=1)
        lines = reference_file.read_text(encoding="utf-8").splitlines()
        texts = [ln.strip() for ln in lines if len(ln.strip()) > 40][:reference_n]
    elif reference_corpus == "stdin":
        console.print("Reading reference texts from stdin (one per line, EOF when done)...")
        for ln in sys.stdin:
            ln = ln.strip()
            if len(ln) > 40:
                texts.append(ln)
            if len(texts) >= reference_n:
                break
    else:
        console.print(f"[red]Unknown reference corpus: {reference_corpus}[/red]")
        raise typer.Exit(code=1)

    if len(texts) < 200:
        console.print(f"[red]Only got {len(texts)} reference texts; need at least 200.[/red]")
        raise typer.Exit(code=1)

    console.print(f"Sending {len(texts)} texts to /init...")
    with _client(timeout=600.0) as c:
        data = _handle_response(c.post("/init", json={"reference_texts": texts}))
    console.print(Panel.fit(
        f"[green]Initialized.[/green]\n"
        f"reference_n: {data['reference_n']}\n"
        f"dim: {data['dim']}",
        title="ccmem init",
    ))


# ---------- add (write) ----------

@app.command()
def add(
    text: Optional[str] = typer.Argument(
        None, help="Text to add (or use --file)."
    ),
    label: Optional[str] = typer.Option(None, "--label", "-l"),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f",
        help="File with one text per blank-line-separated paragraph.",
    ),
):
    """Add a note (or notes from a file) as concept cell(s)."""
    if text is None and file is None:
        console.print("[red]Provide either TEXT or --file.[/red]")
        raise typer.Exit(code=1)

    if file is not None:
        raw = file.read_text(encoding="utf-8")
        # Split by blank line
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        if not paragraphs:
            console.print("[red]No paragraphs found in file.[/red]")
            raise typer.Exit(code=1)
        console.print(f"Adding {len(paragraphs)} paragraphs from {file}...")
        with _client() as c:
            data = _handle_response(c.post(
                "/write_many",
                json={"texts": paragraphs,
                      "labels": [label] * len(paragraphs) if label else None},
            ))
        console.print(f"[green]Added cells:[/green] {data['cell_ids']}")
    else:
        with _client() as c:
            data = _handle_response(c.post(
                "/write", json={"text": text, "label": label},
            ))
        console.print(
            f"[green]Added cell #{data['cell_id']}[/green]"
            + (f" (label: {data['label']})" if data['label'] else "")
        )


# ---------- query ----------

@app.command()
def query(
    text: str = typer.Argument(..., help="Query text."),
    top_k: int = typer.Option(10, "--top-k", "-k"),
    include_silent: bool = typer.Option(
        False, "--include-silent",
        help="Also return non-firing cells, ranked by activation.",
    ),
    no_text: bool = typer.Option(
        False, "--no-text",
        help="Don't print the source text (just IDs and scores).",
    ),
):
    """Query the bank with a text."""
    with _client() as c:
        data = _handle_response(c.post(
            "/query",
            json={"text": text, "top_k": top_k,
                  "include_silent": include_silent},
        ))

    console.print(f"\n[bold]Query:[/bold] {text}")
    console.print(f"[dim]{data['n_hits']} hit(s)[/dim]\n")

    if data["n_hits"] == 0:
        console.print("[yellow]No cells fired.[/yellow]")
        if not include_silent:
            console.print("[dim]Tip: try --include-silent to see near-misses.[/dim]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", justify="right")
    table.add_column("Kind")
    table.add_column("Margin", justify="right")
    table.add_column("Label")
    if not no_text:
        table.add_column("Source / sources")

    for h in data["hits"]:
        margin_str = f"{h['margin']:+.4f}"
        margin_color = "green" if h["margin"] > 0 else "yellow"
        if not no_text:
            if h["source_text"]:
                text_preview = h["source_text"][:140] + (
                    "..." if len(h["source_text"]) > 140 else ""
                )
            elif h["source_cell_ids"]:
                text_preview = f"bound from {h['source_cell_ids']}"
            else:
                text_preview = ""
            table.add_row(
                str(h["cell_id"]),
                h["kind"],
                f"[{margin_color}]{margin_str}[/{margin_color}]",
                h["label"] or "",
                text_preview,
            )
        else:
            table.add_row(
                str(h["cell_id"]),
                h["kind"],
                f"[{margin_color}]{margin_str}[/{margin_color}]",
                h["label"] or "",
            )
    console.print(table)


# ---------- bind ----------

@app.command()
def bind(
    cell_ids: List[int] = typer.Argument(
        ..., help="Two or more cell IDs to bind."
    ),
    label: Optional[str] = typer.Option(None, "--label", "-l"),
):
    """Bind two or more cells into a single composite concept cell."""
    if len(cell_ids) < 2:
        console.print("[red]Need at least 2 cells to bind.[/red]")
        raise typer.Exit(code=1)
    with _client() as c:
        data = _handle_response(c.post(
            "/bind",
            json={"source_cell_ids": cell_ids, "label": label},
        ))
    fires = data["items_fire_after_binding"]
    fire_str = ", ".join(
        f"#{cid}={'✓' if f else '✗'}"
        for cid, f in zip(data["source_cell_ids"], fires)
    )
    color = "green" if all(fires) else "yellow"
    console.print(Panel.fit(
        f"[{color}]Bound cell #{data['bound_cell_id']}[/{color}]\n"
        f"sources: {data['source_cell_ids']}\n"
        f"theta_readout: {data['theta_readout']:.4f}\n"
        f"alignment_to_mean: {data['alignment_to_mean']:.4f}\n"
        f"fires: {fire_str}",
        title="ccmem bind",
    ))


# ---------- list / show / delete ----------

@app.command(name="list")
def list_cells(
    limit: int = typer.Option(50, "--limit"),
    offset: int = typer.Option(0, "--offset"),
    kind: Optional[str] = typer.Option(None, "--kind", help="single | bound"),
):
    """List cells in the bank."""
    with _client() as c:
        params = {"limit": limit, "offset": offset}
        if kind:
            params["kind"] = kind
        data = _handle_response(c.get("/cells", params=params))

    if not data["cells"]:
        console.print("[yellow]No cells in bank.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("ID", justify="right")
    table.add_column("Kind")
    table.add_column("Theta", justify="right")
    table.add_column("Label")
    table.add_column("Source / sources")

    for cell in data["cells"]:
        if cell["source_text"]:
            preview = cell["source_text"][:100] + (
                "..." if len(cell["source_text"]) > 100 else ""
            )
        elif cell["source_cell_ids"]:
            preview = f"bound from {cell['source_cell_ids']}"
        else:
            preview = ""
        table.add_row(
            str(cell["id"]),
            cell["kind"],
            f"{cell['theta']:.3f}",
            cell["label"] or "",
            preview,
        )
    console.print(table)
    console.print(
        f"[dim]Showing {len(data['cells'])} of {data['total']} total.[/dim]"
    )


@app.command()
def show(cell_id: int = typer.Argument(...)):
    """Show full details of a single cell."""
    with _client() as c:
        try:
            r = c.get(f"/cells/{cell_id}")
            data = _handle_response(r)
        except typer.Exit:
            raise
    console.print(Panel.fit(
        f"[bold]ID:[/bold] {data['id']}\n"
        f"[bold]Kind:[/bold] {data['kind']}\n"
        f"[bold]Label:[/bold] {data['label'] or '(none)'}\n"
        f"[bold]Theta:[/bold] {data['theta']:.4f}\n"
        f"[bold]Created at:[/bold] {data['created_at']}\n"
        + (f"[bold]Source text:[/bold]\n{data['source_text']}\n"
           if data["source_text"] else "")
        + (f"[bold]Source cells:[/bold] {data['source_cell_ids']}\n"
           if data["source_cell_ids"] else ""),
        title=f"cell #{cell_id}",
    ))


@app.command()
def delete(
    cell_id: int = typer.Argument(...),
    yes: bool = typer.Option(False, "--yes", "-y"),
):
    """Delete a cell."""
    if not yes:
        confirm = typer.confirm(f"Delete cell #{cell_id}?")
        if not confirm:
            raise typer.Exit(code=1)
    with _client() as c:
        data = _handle_response(c.delete(f"/cells/{cell_id}"))
    if data["deleted"]:
        console.print(f"[green]Deleted cell #{cell_id}.[/green]")
    else:
        console.print(f"[yellow]Could not delete #{cell_id}:[/yellow] {data['reason']}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
