import umap
from helpers import word_wrap, project_embeddings
from pypdf import PdfReader
import os
import openai
from dotenv import load_dotenv
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import chromadb
from sentence_transformers import SentenceTransformer
import numpy as np

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OpenAI API key not found. Please check your .env file or environment variables.")
openai.api_key = openai_key

# Use the model that outputs 768-dimensional embeddings
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Function to generate embeddings using the SentenceTransformer model
def embedding_function(texts):
    embeddings = model.encode(texts)
    # Verify the embedding dimensions are 768 (should print [num_texts, 768])
    print(f"Embedding shape for input: {np.array(embeddings).shape}")
    return embeddings.tolist()  # Convert NumPy arrays to lists

# Step 1: Read and split the PDF content
pdf_path = r'C:\Users\simon\Desktop\Report\microsoft-annual-report.pdf'
reader = PdfReader(pdf_path)

# Extract text from each page of the PDF
pdf_texts = [p.extract_text().strip() if p.extract_text() else '' for p in reader.pages]
pdf_texts = [text for text in pdf_texts if text]  # Filter out empty strings

# Split text using the character splitter
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

# Further split text into tokens for more manageable chunks
token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# Step 2: Initialize Chroma Client
chroma_client = chromadb.Client()

# Delete the existing collection to avoid dimension mismatch issues
try:
    chroma_client.delete_collection(name="microsoft-collection")
    print("Deleted the existing collection.")
except Exception as e:
    print(f"Collection not found or could not be deleted: {e}")

# Create a new Chroma collection
chroma_collection = chroma_client.create_collection("microsoft-collection")

# Step 3: Generate embeddings for documents (ensure 768 dimensions)
document_ids = [str(i) for i in range(len(token_split_texts))]
document_embeddings = embedding_function(token_split_texts)

# Verify the shape of document embeddings (should be [num_docs, 768])
print(f"Document embeddings shape: {np.array(document_embeddings).shape}")

# Add the documents and their embeddings to the Chroma collection
chroma_collection.add(ids=document_ids, documents=token_split_texts, embeddings=document_embeddings)

# Ensure the documents were added successfully
print(f"Total documents in collection: {chroma_collection.count()}")

# Step 4: Query the collection (ensure query embedding has 768 dimensions)
query = "What was the total revenue for the year?"

# Generate the embedding for the query
query_embedding = embedding_function([query])[0]  # Embed the query

# Check if the query embedding has the correct shape (768 dimensions)
print(f"Query embedding shape: {np.array(query_embedding).shape}")

# Perform the query on the Chroma collection
query_results = chroma_collection.query(query_embeddings=[query_embedding], n_results=5)

# Retrieve and display the documents from the query results
retrieved_documents = query_results["documents"][0]
for document in retrieved_documents:
    print(word_wrap(document))
    print("\n")

# Step 5: Generate augmented queries using OpenAI
def generate_multi_query(query, model="gpt-3.5-turbo"):
    prompt = """
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. 
    For the given question, propose up to five related questions to assist them in finding the information they need. 
    Provide concise, single-topic questions (without compounding sentences) that cover various aspects of the topic. 
    Ensure each question is complete and directly related to the original inquiry. 
    List each question on a separate line without numbering.
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=200  # Adjust as needed
    )

    # Extract content and split into multiple queries
    content = response['choices'][0]['message']['content']
    queries = content.split("\n")
    queries = [q.strip() for q in queries if q.strip()]  # Filter out empty lines
    return queries

# Generate augmented queries
original_query = "What details can you provide about the factors that led to revenue growth?"
aug_queries = generate_multi_query(original_query)

# Print the generated queries
for query in aug_queries:
    print(f"Generated query: {query}")

# Step 6: Combine original and augmented queries
joint_query = [original_query] + aug_queries  # Combine into a list

# Embed all queries to ensure they are 768-dimensional
joint_query_embeddings = embedding_function(joint_query)

# Query the collection with the joint queries
joint_results = chroma_collection.query(
    query_embeddings=joint_query_embeddings, n_results=5
)
retrieved_documents = joint_results["documents"]

# Deduplicate the retrieved documents
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)

# Print the deduplicated results
for document in unique_documents:
    print(word_wrap(document))
    print("-" * 100)

# Step 7: Perform UMAP dimensionality reduction
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)

projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)
projected_query_embedding = project_embeddings([query_embedding], umap_transform)
projected_augmented_queries = project_embeddings(joint_query_embeddings, umap_transform)

# Visualize results
import matplotlib.pyplot as plt

# Plot the projected embeddings
plt.figure()
plt.scatter(projected_dataset_embeddings[:, 0], projected_dataset_embeddings[:, 1], s=10, color="gray")
plt.scatter(projected_query_embedding[:, 0], projected_query_embedding[:, 1], s=150, marker="X", color="r")
plt.scatter(projected_augmented_queries[:, 0], projected_augmented_queries[:, 1], s=150, marker="X", color="orange")
plt.gca().set_aspect("equal", "datalim")
plt.title(f"Projection for Query: {original_query}")
plt.axis("off")
plt.show()
