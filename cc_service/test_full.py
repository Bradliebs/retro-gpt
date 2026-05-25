"""Broader test suite hitting many domains, binding, and edge cases."""
import httpx
import time

client = httpx.Client(base_url="http://127.0.0.1:8765", timeout=60)

def show(label, hits, n=3):
    print(f"\n=== {label} ===")
    if not hits:
        print("  (no hits)")
        return
    for i, h in enumerate(hits[:n], 1):
        lbl = (h.get("label") or "(no label)")[:70]
        print(f"  {i}. [{h['activation']:.3f}] {lbl}")

def q(text, top_k=5, include_silent=True):
    return client.post("/query", json={"text": text, "top_k": top_k, "include_silent": include_silent}).json()["hits"]

info = client.get("/info").json()
print(f"Bank: {info['n_cells']:,} cells | dim={info['dim']}\n")

# === 1. Cross-domain knowledge ===
print("\n" + "="*60)
print("TEST 1: Cross-domain knowledge retrieval")
print("="*60)
cross_domain = [
    "what causes earthquakes",
    "how do vaccines work",
    "the French Revolution",
    "what is photosynthesis",
    "how does a nuclear reactor work",
    "what is the theory of relativity",
    "the fall of the Roman Empire",
    "how do black holes form",
    "Shakespeare's most famous plays",
    "the human immune system",
]
for query in cross_domain:
    hits = q(query, top_k=1)
    if hits:
        h = hits[0]
        lbl = (h.get("label") or "(no label)")[:60]
        print(f"  [{h['activation']:.3f}] {query:<45} -> {lbl}")
    else:
        print(f"  [-----] {query:<45} -> NO HITS")

# === 2. Coding queries ===
print("\n" + "="*60)
print("TEST 2: Programming and computer science")
print("="*60)
coding = [
    "implement quicksort algorithm",
    "what is a hash table",
    "dynamic programming explained",
    "SQL join types",
    "REST API design principles",
    "what is dependency injection",
    "explain CAP theorem",
    "TCP versus UDP",
    "git rebase versus merge",
    "what is functional programming",
]
for query in coding:
    hits = q(query, top_k=1)
    if hits:
        h = hits[0]
        lbl = (h.get("label") or "(no label)")[:60]
        print(f"  [{h['activation']:.3f}] {query:<45} -> {lbl}")
    else:
        print(f"  [-----] {query:<45} -> NO HITS")

# === 3. Disambiguation (same word, different meanings) ===
print("\n" + "="*60)
print("TEST 3: Disambiguation — does context steer results?")
print("="*60)
ambig_pairs = [
    ("python the programming language", "python the snake"),
    ("apple the fruit", "apple the company"),
    ("java the island", "java the programming language"),
    ("mercury the planet", "mercury the chemical element"),
    ("a bank for storing money", "the bank of a river"),
]
for a, b in ambig_pairs:
    ha = q(a, top_k=1)
    hb = q(b, top_k=1)
    la = (ha[0].get("label") or "(none)")[:45] if ha else "NO HITS"
    lb = (hb[0].get("label") or "(none)")[:45] if hb else "NO HITS"
    print(f"  '{a}' -> {la}")
    print(f"  '{b}' -> {lb}")
    print()

# === 4. Top-k diversity ===
print("\n" + "="*60)
print("TEST 4: Top-5 results for an open query")
print("="*60)
hits = q("machine learning neural networks", top_k=5)
show("'machine learning neural networks'", hits, n=5)

# === 5. Activation distribution ===
print("\n" + "="*60)
print("TEST 5: Activation profile across queries")
print("="*60)
profile_queries = [
    "Albert Einstein",
    "World War II",
    "DNA",
    "the Great Wall of China",
    "quantum mechanics",
]
for query in profile_queries:
    hits = q(query, top_k=10)
    if hits:
        top = hits[0]["activation"]
        avg = sum(h["activation"] for h in hits) / len(hits)
        bottom = hits[-1]["activation"]
        print(f"  '{query:<30}' top={top:.3f} avg={avg:.3f} bot={bottom:.3f}  ({len(hits)} hits)")

# === 6. Binding test ===
print("\n" + "="*60)
print("TEST 6: Hebbian binding — fuse two concepts")
print("="*60)
hits_a = q("solar system planets", top_k=1)
hits_b = q("Jupiter the planet", top_k=1)
if hits_a and hits_b and hits_a[0]["cell_id"] != hits_b[0]["cell_id"]:
    a_id = hits_a[0]["cell_id"]
    b_id = hits_b[0]["cell_id"]
    print(f"  Cell A: [{hits_a[0]['activation']:.3f}] {(hits_a[0].get('label') or '?')[:55]}")
    print(f"  Cell B: [{hits_b[0]['activation']:.3f}] {(hits_b[0].get('label') or '?')[:55]}")
    bind_resp = client.post("/bind", json={
        "source_cell_ids": [a_id, b_id],
        "label": "[test] solar system + jupiter",
    }).json()
    bound_id = bind_resp["bound_cell_id"]
    print(f"  -> bound cell {bound_id}")
    # Query toward the bound concept
    hits_c = q("the largest planet orbiting our sun", top_k=5)
    show("After binding: 'largest planet orbiting our sun'", hits_c, n=5)
    found = next((h for h in hits_c if h["cell_id"] == bound_id), None)
    if found:
        rank = hits_c.index(found) + 1
        print(f"  ✓ Bound cell ranked #{rank} with activation {found['activation']:.3f}")
    else:
        print(f"  ✗ Bound cell not in top 5")
    client.delete(f"/cells/{bound_id}")
    print(f"  cleaned up bound cell")
else:
    print("  could not find two distinct cells for binding")

# === 7. Threshold behavior ===
print("\n" + "="*60)
print("TEST 7: Nonsense and edge-case queries")
print("="*60)
weird = [
    "asdfqwerty zxcvbnm",
    "the quick brown fox jumps over the lazy dog",
    "1234567890",
    "",
    "a",
]
for query in weird:
    try:
        hits = q(query, top_k=3, include_silent=True)
        if hits:
            top = hits[0]
            lbl = (top.get("label") or "(no label)")[:50]
            print(f"  '{query[:30]:<30}' -> [{top['activation']:.3f}] {lbl}")
        else:
            print(f"  '{query[:30]:<30}' -> no hits")
    except httpx.HTTPStatusError as e:
        print(f"  '{query[:30]:<30}' -> HTTP {e.response.status_code}")
    except Exception as e:
        print(f"  '{query[:30]:<30}' -> {type(e).__name__}: {e}")

# === 8. Latency ===
print("\n" + "="*60)
print("TEST 8: Query latency (GPU-encoded)")
print("="*60)
queries = ["history of Rome", "binary search", "evolution of species", "the periodic table", "machine learning"] * 4
times = []
for query in queries:
    t0 = time.perf_counter()
    client.post("/query", json={"text": query, "top_k": 10, "include_silent": True})
    times.append((time.perf_counter() - t0) * 1000)
times.sort()
print(f"  n={len(times)}  min={times[0]:.1f}ms  median={times[len(times)//2]:.1f}ms  p95={times[int(len(times)*0.95)]:.1f}ms  max={times[-1]:.1f}ms")

print("\n" + "="*60)
final = client.get("/info").json()
print(f"FINAL Bank: {final['n_cells']:,} cells")
print("="*60)
