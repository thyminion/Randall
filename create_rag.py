import chromadb
from sentence_transformers import SentenceTransformer

DISTANCE_METRIC = "cosine"

statements = [
    line.strip()
    for line in open("facts.txt", encoding="utf-8") # one fact per line
    if line.strip()
]

model = SentenceTransformer("all-MiniLM-L6-v2") # trained neural net, tokenizes each fact, the via trained parameters creates embeddings
embeddings = model.encode(statements).tolist()

client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_or_create_collection(
    f"facts_{DISTANCE_METRIC}", # collection name
    metadata={"hnsw:space": DISTANCE_METRIC},
)
collection.upsert(
    ids=[str(index) for index in range(len(statements))],
    documents=statements,
    embeddings=embeddings, #384 dimensions
)

print("First fact:", statements[0])
print("Embedding:", embeddings[0])