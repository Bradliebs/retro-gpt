import httpx, os
c = httpx.Client(base_url='http://127.0.0.1:8765', timeout=10)
i = c.get('/info').json()
db = r'h:\MiniLM\cc_service\bank.db'
sz = os.path.getsize(db) / (1024**3)
print(f"cells={i['n_cells']} dim={i['dim']} db_gb={sz:.2f}")
