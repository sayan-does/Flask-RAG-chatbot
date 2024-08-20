import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Initialize the ChromaDB client with the correct settings
client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",  # Use the correct Chroma DB implementation
    #tried an earlier version but that was deprecated
    persist_directory="./chromadb_data"  # Directory to store the ChromaDB data
))

# Create a new collection
collection_name = "new_pdf_text_collection"
collection = client.create_collection(name=collection_name)

# Initialize the sentence transformer model with CPU only
model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def chunk_text(text, chunk_size=500):
    """Split the text into smaller chunks."""
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def add_text_to_chroma(collection, text, chunk_size=500):
    chunks = chunk_text(text, chunk_size)
    for i, chunk in enumerate(chunks):
        # Embed each chunk of text
        embedding = model.encode(chunk).tolist()
        # Add to ChromaDB
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"doc_{i}"]
        )

# Load your extracted text from the .txt file
with open("output_text_file.txt", "r", encoding="utf-8") as file:
    pdf_text = file.read()

# Add the text to ChromaDB
add_text_to_chroma(collection, pdf_text)

print(f"Text successfully added to ChromaDB in collection '{collection_name}'!")

# Example query
query_text = "What is the average cost of plumbing services?"

# Embed the query
query_embedding = model.encode(query_text).tolist()

# Query the ChromaDB
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3  # Number of top results to return
)

# Print the results
for result in results['documents'][0]:
    print("Relevant text:", result)    #user query one by one
