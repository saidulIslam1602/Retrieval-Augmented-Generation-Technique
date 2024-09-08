import os
import openai  # Correct OpenAI import
from dotenv import load_dotenv
from helpers import word_wrap, load_chroma
from pypdf import PdfReader
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
from sentence_transformers import CrossEncoder

# Load environment variables from .env file
load_dotenv()

# Ensure the OpenAI API key is loaded
openai_key = os.getenv("OPENAI_API_KEY")
if not openai_key:
    raise ValueError("OpenAI API key not found. Please check your .env file or environment variables.")
openai.api_key = openai_key  # Corrected OpenAI usage

# Initialize the SentenceTransformer model to generate embeddings
model = SentenceTransformer('all-mpnet-base-v2')  # Model produces 768-dimensional embeddings

# Load PDF and extract text
pdf_path = r'C:\Users\simon\Desktop\Report\microsoft-annual-report.pdf'
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found at {pdf_path}")

reader = PdfReader(pdf_path)
pdf_texts = [p.extract_text().strip() for p in reader.pages]

# Filter empty strings from the extracted texts
pdf_texts = [text for text in pdf_texts if text]
if not pdf_texts:
    raise ValueError("No valid text extracted from the PDF.")

# Text splitting
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# Initialize ChromaDB client
chroma_client = chromadb.Client()

# Create or get the Chroma collection (without embedding function)
chroma_collection = chroma_client.get_or_create_collection("microsoft-collect")

# Add documents to the Chroma collection and generate embeddings
ids = [str(i) for i in range(len(token_split_texts))]
if not ids:
    raise ValueError("No valid chunks of text to add to the Chroma collection.")

# Generate embeddings for the tokenized text chunks
embeddings = model.encode(token_split_texts).tolist()

# Add the documents and their embeddings to the Chroma collection
chroma_collection.add(ids=ids, documents=token_split_texts, embeddings=embeddings)

# Check the document count in the Chroma collection
count = chroma_collection.count()
print(f"Total documents in collection: {count}")

# Define the query and perform the Chroma query
query = "What has been the investment in research and development?"

# Generate the embedding for the query
query_embedding = model.encode([query]).tolist()

# Perform the query in the Chroma collection
results = chroma_collection.query(query_embeddings=query_embedding, n_results=10)

# Ensure there are results returned from Chroma
if not results or "documents" not in results or not results["documents"]:
    raise ValueError("No documents were retrieved from the Chroma collection.")

# Retrieve and display the top documents
retrieved_documents = results["documents"][0]
if not retrieved_documents:
    raise ValueError("No documents were retrieved for the given query.")
for document in retrieved_documents:
    print(word_wrap(document))
    print("")

# Use a cross-encoder for reranking the retrieved documents
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[query, doc] for doc in retrieved_documents]
scores = cross_encoder.predict(pairs)

# Display the scores and ranking
print("Scores:")
for score in scores:
    print(score)

print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o + 1)

# Generate new queries for further search
original_query = (
    "What were the most important factors that contributed to increases in revenue?"
)
generated_queries = [
    "What were the major drivers of revenue growth?",
    "Were there any new product launches that contributed to the increase in revenue?",
    "Did any changes in pricing or promotions impact the revenue growth?",
    "What were the key market trends that facilitated the increase in revenue?",
    "Did any acquisitions or partnerships contribute to the revenue growth?",
]

# Concatenate the original query with the generated queries
queries = [original_query] + generated_queries

# Generate embeddings for all queries
query_embeddings = model.encode(queries).tolist()

# Perform another Chroma query
results = chroma_collection.query(query_embeddings=query_embeddings, n_results=10)

# Deduplicate the retrieved documents
retrieved_documents = results["documents"]
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)

# Prepare pairs for cross-encoder reranking
unique_documents = list(unique_documents)
pairs = [[original_query, doc] for doc in unique_documents]
scores = cross_encoder.predict(pairs)

print("Scores:")
for score in scores:
    print(score)

print("New Ordering:")
for o in np.argsort(scores)[::-1]:
    print(o)

# Select top documents based on cross-encoder scores
top_indices = np.argsort(scores)[::-1][:5]
top_documents = [unique_documents[i] for i in top_indices]
context = "\n\n".join(top_documents)

# Generate the final answer using OpenAI's GPT model
def generate_multi_query(query, context, model="gpt-3.5-turbo"):
    prompt = f"""
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about an annual report. Based on the following context:\n\n{context}\n\nAnswer the query: '{query}'.
    """
    messages = [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": f"Answer the query: '{query}'",
        },
    ]

    # Correct OpenAI API usage
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
    )
    content = response['choices'][0]['message']['content']
    return content

# Get the final answer
res = generate_multi_query(query=original_query, context=context)
print("Final Answer:")
print(res)
