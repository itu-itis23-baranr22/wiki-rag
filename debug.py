from src.vector_store import WikiVectorStore

s = WikiVectorStore()
print(f"Toplam chunk: {s.count()}\n")

# 1) Einstein adıyla doğrudan ara
print("--- Sorgu: 'Albert Einstein' ---")
for r in s.query("Albert Einstein", top_k=5, type_filter="person"):
    e = r["metadata"]["entity"]
    d = r["distance"]
    snippet = r["document"][:90].replace("\n", " ")
    print(f"  dist={d:.3f} | {e:25} | {snippet}")

# 2) Daha doğal bir sorguyla ara
print("\n--- Sorgu: 'Who was Albert Einstein?' ---")
for r in s.query("Who was Albert Einstein?", top_k=5, type_filter="person"):
    e = r["metadata"]["entity"]
    d = r["distance"]
    snippet = r["document"][:90].replace("\n", " ")
    print(f"  dist={d:.3f} | {e:25} | {snippet}")

# 3) Filtresiz ara (her tip)
print("\n--- Sorgu: 'Albert Einstein physicist relativity' (filtresiz) ---")
for r in s.query("Albert Einstein physicist relativity", top_k=5):
    e = r["metadata"]["entity"]
    t = r["metadata"]["type"]
    d = r["distance"]
    snippet = r["document"][:90].replace("\n", " ")
    print(f"  dist={d:.3f} | {t:6} | {e:25} | {snippet}")