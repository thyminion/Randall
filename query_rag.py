import chromadb
from sentence_transformers import SentenceTransformer


INPUT_TEXT = "What does the agreement say about conflicts of interest?"
DISTANCE_METRIC = "cosine"

model = SentenceTransformer("all-MiniLM-L6-v2")
query_embedding = model.encode(INPUT_TEXT).tolist()

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_collection(f"facts_{DISTANCE_METRIC}")
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=1,
)

top_fact = results["documents"][0][0]
messages = [
    {"role": "system", "content": f"Use this retrieved fact to answer the user: {top_fact}"},
    {"role": "user", "content": INPUT_TEXT},
]

print(messages)
cosine_similarity = 1 - results["distances"][0][0]
print("Cosine similarity:", cosine_similarity)

# 1: same direction, highest similarity
# 0: orthogonal, little directional similarity−
# −1: opposite directions