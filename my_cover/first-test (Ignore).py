from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import (RecursiveJsonSplitter, SentenceTransformersTokenTextSplitter)
import json
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import chromadb


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build full path to your JSON file
file_path = os.path.join(BASE_DIR, "json", "schema.json")

with open(file_path, "r", encoding="utf-8") as file:
    db = json.load(file)


#RecursiveJsonSplitter only works with dictionary, so it is important to fetch the json raw
json_splitter = RecursiveJsonSplitter(
    max_chunk_size = 500,
    min_chunk_size= 200
    
)
json_chunks = json_splitter.split_json(json_data=db)

#Convert json dictionary to a string so that it can be split further to token chunks
json_text = [json.dumps(chunk, ensure_ascii = False, indent=2) for chunk in json_chunks]

token_chunk = SentenceTransformersTokenTextSplitter(
    tokens_per_chunk=256,
    chunk_overlap=50,
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

# Create proper LangChain Documents with metadata
from langchain_core.documents import Document

documents = []
for i, text in enumerate(json_text):
    doc = Document(
        page_content=text,
        metadata={
            "chunk_index": i,
            "source": file_path,
            "schema_type": "database_schema",
            "industry": "multi_industry"  # You can customize this per schema
        }
    )
    documents.append(doc)

token_split_json = token_chunk.split_documents(documents)

chroma_path = os.path.join(BASE_DIR, "chroma_db")

chroma_client = chromadb.PersistentClient(path=chroma_path)
embedding_function = SentenceTransformerEmbeddingFunction(model_name= "sentence-transformers/all-MiniLM-L6-v2")

collection = chroma_client.get_or_create_collection("Microsoft Package", embedding_function=embedding_function, metadata={"hnsw:space": "cosine"})

texts = [doc.page_content for doc in token_split_json]
metadatas =[doc.metadata for doc in token_split_json]
ids =[f"chunk_schema_{i}" for i in range(len(token_split_json))]

#collection.add(ids=ids, documents=texts, mnetadatas=metadatas)
#Add to collection in batches or large schema
batch_size = 100
for i in range(0, len(texts), batch_size):
    batch_end = min(i + batch_size, len(texts))
    collection.add(
        ids=ids[i:batch_end],
        documents=texts[i:batch_end],
        metadatas=metadatas[i:batch_end]
    )


user_query = "Find all savings accounts with balance greater than 1,000,000."

emb_query = collection.query(query_texts=[user_query], n_results=5, include =["documents", "metadatas", "distances"])

#Make the context clear to the LLM
contexts = []
for doc, meta, dist in zip(emb_query["documents"][0], emb_query["metadatas"][0],   emb_query["distances"][0]):

    contexts.append({
            "content": doc,
            "relevance_score": 1 - dist,  # Convert distance to similarity
            "chunk_index": meta.get("chunk_index", "unknown")
    })

# Build structured context string for prompt
context_str = "\n\n---\n\n".join([
    f"Schema Chunk {ctx['chunk_index']} (Relevance: {ctx['relevance_score']:.2f}):\n{ctx['content']}"
    for ctx in contexts
])


API_KEY = os.getenv("API_KEY")
client = genai.Client(api_key=API_KEY) 

def generate_answer(query:str, model="gemini-3-flash-preview"):
    context=context_str
    prompt= f"""You are an expert Database Architect and SQL Engineer with deep expertise in multiple industries including banking, oil & gas, insurance, and government systems.

Your task is to generate precise, syntactically correct SQL queries based on the provided database schema context.

## SCHEMA CONTEXT:
The following JSON schema chunks represent the database structure most relevant to the user's query: {context}

## CRITICAL INSTRUCTIONS:
1. **Schema Adherence**: Use ONLY tables, columns, and relationships present in the provided schema context. Never hallucinate column names.
2. **Table Selection**: Identify the most relevant tables from the schema chunks above. If multiple tables are needed, use appropriate JOINs.
3. **Column Mapping**: Map user query concepts to actual column names in the schema. Pay attention to data types.
4. **Query Type**: Determine if the query requires SELECT, INSERT, UPDATE, DELETE, or CREATE statements.
5. **Filtering**: Include all WHERE clauses, GROUP BY, HAVING, and ORDER BY as implied by the user query.
6. **Aggregation**: Use appropriate aggregate functions (COUNT, SUM, AVG, etc.) when needed.
7. **Aliases**: Use clear table aliases (e.g., c for customers, o for orders) for readability.
8. **Comments**: Add brief SQL comments (-- ) explaining complex joins or filters.

## OUTPUT FORMAT:
Provide your response in this exact format:

**Analysis:**
Brief analysis of what tables/columns are needed and why.

**SQL Query:**
```sql
-- Your SQL query here
"""
    

    response = client.models.generate_content(
        model = model,
        contents=query,
        config= types.GenerateContentConfig(
        temperature = 0.3,
        top_p=0.95,
        system_instruction=prompt,
        
        )
        
    )

    result = response.text

    return result

answer = generate_answer(user_query)

combined_query = f"{user_query}{answer}"




