from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "json", "schema.json")

with open(file_path, "r", encoding="utf-8") as file:
    db = json.load(file)

# ==========================================
# ⭐ TABLE-LEVEL CHUNKING (NEW)
# ==========================================

def table_to_rich_text(database_name, table):
    """Convert a table schema into rich natural text for embeddings."""
    
    lines = [
        f"Database: {database_name}",
        f"Table: {table['table_name']}",
        "Columns:"
    ]
    
    for col in table["columns"]:
        pk = " PRIMARY KEY" if col.get("is_primary_key") else ""
        nullable = " NULLABLE" if col.get("is_nullable") else " NOT NULL"
        
        lines.append(
            f"- {col['column_name']} ({col['data_type']}){pk}{nullable}"
        )
    
    return "\n".join(lines)


texts = []
metadatas = []
ids = []

chunk_index = 0

for database in db["databases"]:
    db_name = database["database_name"]
    
    for table in database["tables"]:
        
        text = table_to_rich_text(db_name, table)
        
        texts.append(text)
        
        metadatas.append({
            "database": db_name,
            "table": table["table_name"],
            "chunk_index": chunk_index,
            "schema_type": "table_definition",
            "source": file_path
        })
        
        ids.append(f"table_{chunk_index}")
        chunk_index += 1


# ==========================================
# CHROMA INITIALIZATION
# ==========================================

chroma_path = os.path.join(BASE_DIR, "chroma_db")
chroma_client = chromadb.PersistentClient(path=chroma_path)

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    "bank_schema_collection",
    embedding_function=embedding_function,
    metadata={"hnsw:space": "cosine"}
)

# Add table chunks (batch-safe)
batch_size = 100
for i in range(0, len(texts), batch_size):
    batch_end = min(i + batch_size, len(texts))
    
    collection.add(
        ids=ids[i:batch_end],
        documents=texts[i:batch_end],
        metadatas=metadatas[i:batch_end]
    )


# ==========================================
# QUERY EXPANSION (UNCHANGED)
# ==========================================

API_KEY = os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY)

def expand_queries(original_query: str) -> list:
    
    expansion_prompt = f"""You are a database schema search expert. Given a user question about data retrieval, generate 3 specific search queries to find relevant database schema information.

Original user question: "{original_query}"

Generate 3 search queries that target different aspects:
1. Table names and structures mentioned or implied
2. Column names, data types, and constraints
3. Relationships, foreign keys, and join conditions

Return ONLY a JSON array of 3 strings."""

    try:
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=expansion_prompt,
            config=types.GenerateContentConfig(temperature=0.3)
        )
        
        text = response.text.strip()
        
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
            
        expanded = json.loads(text)
        
        if isinstance(expanded, list) and len(expanded) > 0:
            return [original_query] + expanded[:3]
            
    except Exception as e:
        print(f"Query expansion failed: {e}")
    
    return [original_query]


user_query = "Show the top 5 customers with the highest total account balance across all their accounts."

all_queries = expand_queries(user_query)
print(f"Expanded into {len(all_queries)} queries: {all_queries}")


# ==========================================
# RETRIEVAL
# ==========================================

all_results = []

for q in all_queries:
    results = collection.query(
        query_texts=[q],
        n_results=5,  # Increased for better coverage
        include=["documents", "metadatas", "distances"]
    )
    
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        all_results.append({
            "content": doc,
            "metadata": meta,
            "distance": dist,
            "source_query": q
        })


# Deduplicate by chunk_index
unique_chunks = {}

for result in all_results:
    chunk_id = result["metadata"]["chunk_index"]
    
    if chunk_id not in unique_chunks or result["distance"] < unique_chunks[chunk_id]["distance"]:
        unique_chunks[chunk_id] = result


sorted_chunks = sorted(unique_chunks.values(), key=lambda x: x["distance"])[:5]


contexts = []

for result in sorted_chunks:
    contexts.append({
        "content": result["content"],
        "relevance_score": 1 - result["distance"],
        "chunk_index": result["metadata"]["chunk_index"],
        "found_by": result["source_query"]
    })


print(f"Retrieved {len(contexts)} unique schema chunks")


context_str = "\n\n---\n\n".join([
    f"Schema Chunk {ctx['chunk_index']} (Relevance: {ctx['relevance_score']:.2f}, Found by: '{ctx['found_by']}'):\n{ctx['content']}"
    for ctx in contexts
])


# ==========================================
# GENERATION (UNCHANGED)
# ==========================================

def generate_answer(query: str, model="gemini-3-flash-preview"):
    
    prompt = f"""You are an expert Database Architect and SQL Engineer.

Use ONLY the schema context below.

SCHEMA CONTEXT:
{context_str}

Prefer returning real-world entities (e.g., customers)."""

    response = client.models.generate_content(
        model=model,
        contents=query,
        config=types.GenerateContentConfig(
            temperature=0.3,
            system_instruction=prompt
        )
    )

    return response.text


answer = generate_answer(user_query)

print(f"AI: {answer}")