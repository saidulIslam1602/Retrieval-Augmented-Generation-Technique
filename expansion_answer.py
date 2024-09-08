import chromadb
import openai
import umap.umap_ as umap  # Correct UMAP import
from pypdf import PdfReader
import os
from dotenv import load_dotenv
from expansion_queries import embedding_function  # Ensure this exists
from helpers import word_wrap, project_embeddings
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from sentence_transformers import SentenceTransformer
import numpy as np  # Import numpy for array handling

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")  # Correct way to set the API key

# 1. Read the PDF file
pdf_path = r'C:\Users\simon\Desktop\Report\microsoft-annual-report.pdf'
reader = PdfReader(pdf_path)

# Extract and clean text from each page, with logging
pdf_texts = []
for page_num, page in enumerate(reader.pages):
    text = page.extract_text().strip()
    if text:
        pdf_texts.append(text)
    print(f"Extracted text from page {page_num + 1}")

# 2. Check if text extraction worked
if not pdf_texts:
    raise ValueError("No text was extracted from the PDF. Please check the PDF extraction process.")

# Initialize text splitters
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))
print(f"Number of chunks after character splitting: {len(character_split_texts)}")

# Check for empty chunks after splitting
character_split_texts = [text for text in character_split_texts if text.strip()]
if not character_split_texts:
    raise ValueError("Text splitting failed. No valid chunks were produced.")

# Token splitter
token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

print(f"Number of chunks after token splitting: {len(token_split_texts)}")

# 3. Load the sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the custom embedding function
class EmbeddingFunction:
    def __call__(self, input):
        return model.encode(input, convert_to_tensor=False).tolist()  # Ensure compatibility with Chroma

# Initialize Chroma and manage collection (delete if it exists, then recreate)
chroma_client = chromadb.Client()

collection_name = "microsoft-collection"

# Delete the collection if it exists
try:
    chroma_client.delete_collection(name=collection_name)
except Exception as e:
    print(f"Collection {collection_name} not found or couldn't be deleted: {e}")

# Create the collection after deleting
chroma_collection = chroma_client.create_collection(
    collection_name, embedding_function=EmbeddingFunction()
)

# Add documents to the Chroma collection
for i, text in enumerate(token_split_texts):
    chroma_collection.add(
        documents=[text],
        ids=[f"doc_{i}"]
    )
    print(f"Added document {i} to the collection.")

# Check the number of documents in the collection
num_documents = len(chroma_collection.get(include=["documents"])["documents"])
print(f"Number of documents in the collection: {num_documents}")

# Retrieve embeddings from Chroma collection
collection_data = chroma_collection.get(include=["embeddings", "documents"])
embeddings = collection_data["embeddings"]

# 4. Check if embeddings are empty or not
if len(embeddings) == 0:
    raise ValueError("Embeddings array is empty. Please ensure documents were embedded properly.")
else:
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Sample embedding: {embeddings[0]}")

# Ensure embeddings are in 2D format
embeddings = np.array(embeddings)
if len(embeddings.shape) == 1:
    embeddings = embeddings.reshape(1, -1)

# UMAP dimensionality reduction
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

# Continue with embedding retrieval and plotting
original_query = "What was the total profit for the year, and how does it compare to the previous year?"
joint_query = f"{original_query}"

results = chroma_collection.query(
    query_texts=[joint_query], n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"][0]

retrieved_embeddings = results["embeddings"][0]
original_query_embedding = EmbeddingFunction()([original_query])
augmented_query_embedding = EmbeddingFunction()([joint_query])

projected_original_query_embedding = project_embeddings(
    original_query_embedding, umap_transform
)
projected_augmented_query_embedding = project_embeddings(
    augmented_query_embedding, umap_transform
)
projected_retrieved_embeddings = project_embeddings(
    retrieved_embeddings, umap_transform
)

import matplotlib.pyplot as plt

# Plot the projected query and retrieved documents in the embedding space
plt.figure()

plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange"
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"{original_query}")
plt.axis("off")
plt.show()  # display the plot
