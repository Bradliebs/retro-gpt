"""Post-load test suite for the memory bank.

Runs semantic queries across different domains, tests binding,
and reports quality metrics.

Usage:
    python test_loaded.py [--base-url http://127.0.0.1:8765]
"""
from __future__ import annotations

import sys
import time

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()
BASE_URL = "http://127.0.0.1:8765"


def api(client: httpx.Client, method: str, path: str, body=None):
    opts = {"method": method, "url": path}
    if body:
        opts["json"] = body
    r = client.request(**opts)
    r.raise_for_status()
    return r.json()


def test_queries(client: httpx.Client):
    """Run diverse queries and check result quality."""
    test_cases = [
        ("how do black holes form", "Physics / Astronomy"),
        ("photosynthesis converts sunlight into energy", "Biology"),
        ("the speed of light in a vacuum", "Physics"),
        ("democracy and voting systems", "Political Science"),
        ("DNA replication during cell division", "Genetics"),
        ("the French Revolution and its causes", "History"),
        ("machine learning and neural networks", "Computer Science"),
        ("climate change and greenhouse gases", "Environment"),
        ("Mozart and classical music composition", "Music"),
        ("the human immune system fights infections", "Medicine"),
    ]

    table = Table(title="Query Quality Test", show_lines=True)
    table.add_column("Domain", style="cyan", width=18)
    table.add_column("Query", width=40)
    table.add_column("Hits", justify="right", width=5)
    table.add_column("Top Score", justify="right", width=10)
    table.add_column("Top Hit Label", width=35)
    table.add_column("Relevant?", justify="center", width=9)

    results = []
    for query_text, domain in test_cases:
        try:
            data = api(client, "POST", "/query", {
                "text": query_text, "top_k": 5, "include_silent": True
            })
            hits = data["hits"]
            n_hits = len(hits)
            top_score = hits[0]["activation"] if hits else 0.0
            top_label = (hits[0]["label"] or "(unlabelled)")[:35] if hits else "-"
            # Heuristic: relevant if top score > 0.3 and we got hits
            relevant = "✓" if top_score > 0.3 and n_hits > 0 else "✗"
            table.add_row(domain, query_text[:40], str(n_hits),
                         f"{top_score:.3f}", top_label, relevant)
            results.append((domain, n_hits, top_score, relevant == "✓"))
        except Exception as e:
            table.add_row(domain, query_text[:40], "ERR", "-", str(e)[:35], "✗")
            results.append((domain, 0, 0.0, False))

    console.print(table)

    # Summary
    total = len(results)
    relevant = sum(1 for _, _, _, r in results if r)
    avg_score = sum(s for _, _, s, _ in results) / total if total else 0
    avg_hits = sum(h for _, h, _, _ in results) / total if total else 0
    console.print(f"\n  Relevant: [green]{relevant}/{total}[/green] | "
                  f"Avg top score: [cyan]{avg_score:.3f}[/cyan] | "
                  f"Avg hits: [cyan]{avg_hits:.1f}[/cyan]")
    return relevant, total


def test_binding(client: httpx.Client):
    """Test binding two related cells together."""
    console.print("\n[bold]Binding Test[/bold]")

    # Query for two related topics
    physics_hits = api(client, "POST", "/query", {
        "text": "gravity and mass", "top_k": 2, "include_silent": True
    })["hits"]
    
    if len(physics_hits) < 2:
        console.print("[yellow]Not enough hits to test binding[/yellow]")
        return False

    id_a = physics_hits[0]["cell_id"]
    id_b = physics_hits[1]["cell_id"]
    label_a = physics_hits[0]["label"] or f"cell-{id_a}"
    label_b = physics_hits[1]["label"] or f"cell-{id_b}"

    console.print(f"  Binding #{id_a} ([cyan]{label_a[:40]}[/cyan]) + "
                  f"#{id_b} ([cyan]{label_b[:40]}[/cyan])")

    try:
        result = api(client, "POST", "/bind", {
            "source_cell_ids": [id_a, id_b],
            "label": "gravity-mass composite (test)"
        })
        console.print(f"  [green]✓ Created bound cell #{result['bound_cell_id']}[/green]")
        console.print(f"    θ={result['theta_readout']:.4f}, "
                      f"alignment={result['alignment_to_mean']:.4f}, "
                      f"fires={result['items_fire_after_binding']}")

        # Query back for the bound concept
        verify = api(client, "POST", "/query", {
            "text": "gravitational force between objects with mass",
            "top_k": 5, "include_silent": True
        })
        bound_in_results = any(h["cell_id"] == result["bound_cell_id"] for h in verify["hits"])
        if bound_in_results:
            console.print("  [green]✓ Bound cell appears in related query results[/green]")
        else:
            console.print("  [yellow]⚠ Bound cell did not appear in top-5 for related query[/yellow]")

        # Clean up test cell
        api(client, "DELETE", f"/cells/{result['bound_cell_id']}")
        console.print(f"  Cleaned up test cell #{result['bound_cell_id']}")
        return True
    except Exception as e:
        console.print(f"  [red]✗ Binding failed: {e}[/red]")
        return False


def test_cross_domain(client: httpx.Client):
    """Test that cross-domain queries return diverse results."""
    console.print("\n[bold]Cross-Domain Diversity Test[/bold]")

    data = api(client, "POST", "/query", {
        "text": "energy transformation and conservation",
        "top_k": 10, "include_silent": True
    })

    labels = set()
    for h in data["hits"]:
        lbl = h["label"] or ""
        labels.add(lbl)

    unique_topics = len(labels)
    console.print(f"  Query: 'energy transformation and conservation'")
    console.print(f"  Hits: {len(data['hits'])}, Unique labels: {unique_topics}")
    for h in data["hits"][:5]:
        console.print(f"    #{h['cell_id']} [{h['activation']:.3f}] {(h['label'] or '(no label)')[:50]}")

    diverse = unique_topics >= 3
    console.print(f"  {'[green]✓ Diverse' if diverse else '[yellow]⚠ Low diversity'} results[/]")
    return diverse


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()

    client = httpx.Client(base_url=args.base_url, timeout=60.0)

    # Check service
    try:
        info = api(client, "GET", "/info")
    except Exception:
        console.print("[red]Cannot connect to service. Is it running?[/red]")
        sys.exit(1)

    console.print(Panel(
        f"Bank: [cyan]{info['n_cells']:,}[/cyan] cells | "
        f"Dim: {info['dim']} | "
        f"Encoder: {info['encoder_model']}",
        title="Memory Bank Status"
    ))

    if info["n_cells"] < 100:
        console.print("[yellow]Warning: Very few cells loaded. Results may be sparse.[/yellow]")

    # Run tests
    start = time.time()
    relevant, total = test_queries(client)
    bind_ok = test_binding(client)
    diverse = test_cross_domain(client)
    elapsed = time.time() - start

    # Final verdict
    console.print(Panel(
        f"Queries: [{'green' if relevant >= 7 else 'yellow'}]{relevant}/{total} relevant[/] | "
        f"Binding: [{'green' if bind_ok else 'red'}]{'PASS' if bind_ok else 'FAIL'}[/] | "
        f"Diversity: [{'green' if diverse else 'yellow'}]{'PASS' if diverse else 'WEAK'}[/] | "
        f"Time: {elapsed:.1f}s",
        title="Test Results",
        border_style="green" if (relevant >= 7 and bind_ok) else "yellow"
    ))


if __name__ == "__main__":
    main()
