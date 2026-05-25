"""Quick test of coding content in the memory bank."""
import httpx

client = httpx.Client(base_url="http://127.0.0.1:8765", timeout=60)
info = client.get("/info").json()
print(f"Bank: {info['n_cells']:,} cells\n")

queries = [
    "how to implement a binary search tree",
    "what is the difference between a list and a tuple in Python",
    "explain the observer design pattern",
    "how does garbage collection work",
    "write a function to reverse a linked list",
    "what is recursion and when to use it",
]

for q in queries:
    data = client.post("/query", json={"text": q, "top_k": 3, "include_silent": True}).json()
    hits = data["hits"]
    if hits:
        top = hits[0]
        score = top["activation"]
        label = (top["label"] or "(no label)")[:65]
        print(f"[{score:.3f}] {q}")
        print(f"   -> {label}")
    else:
        print(f"[-----] {q}")
        print(f"   -> NO HITS")
    print()
