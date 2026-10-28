import requests, json, hashlib, os

CACHE = "llm_cache.json"
_cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}

def ask(prompt, model="qwen2.5:7b"):
    key = hashlib.md5((model + prompt).encode()).hexdigest()
    if key in _cache:
        return _cache[key]
    r = requests.post("http://localhost:11434/api/generate", json={
        "model": model, "prompt": prompt, "format": "json",
        "stream": False, "options": {"temperature": 0}}, timeout=300)
    out = json.loads(r.json()["response"])
    _cache[key] = out
    json.dump(_cache, open(CACHE, "w"))
    return out