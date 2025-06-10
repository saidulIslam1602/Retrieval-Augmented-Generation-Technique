"""
Advanced RAG System with Multiple Sophisticated Features
========================================================

This module implements an advanced Retrieval-Augmented Generation system with:
- Hybrid Search (Dense + Sparse)
- Multi-modal Document Processing
- Adaptive Retrieval Strategies
- Query Classification and Routing
- Advanced Chunking Techniques
- Retrieval Quality Evaluation
- Memory Management
- Real-time Learning
"""

import os
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import pickle
from datetime import datetime
import uuid

# Core ML/NLP Libraries
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.utils import embedding_functions
import umap

# Document Processing
from pypdf import PdfReader
import pytesseract
from PIL import Image
import cv2
import camelot
import tabula

# Text Processing
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    SpacyTextSplitter,
    NLTKTextSplitter
)
from langchain.schema import Document
import spacy
import nltk

# Advanced NLP
from transformers import (
    AutoTokenizer, AutoModel, pipeline,
    DPRQuestionEncoder, DPRContextEncoder,
    DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer
)
import torch

# API and Web
import openai
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import streamlit as st

# Utilities
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of queries for routing to appropriate retrieval strategies"""
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    COMPARISON = "comparison"
    TEMPORAL = "temporal"
    NUMERICAL = "numerical"
    CONCEPTUAL = "conceptual"

class RetrievalStrategy(Enum):
    """Different retrieval strategies"""
    DENSE_ONLY = "dense_only"
    SPARSE_ONLY = "sparse_only"
    HYBRID = "hybrid"
    DPR = "dpr"
    MULTI_VECTOR = "multi_vector"

@dataclass
class RetrievalResult:
    """Container for retrieval results with metadata"""
    documents: List[str]
    scores: List[float]
    metadata: List[Dict[str, Any]]
    strategy_used: str
    query_embedding: Optional[np.ndarray] = None
    retrieval_time: float = 0.0

@dataclass
class QueryAnalysis:
    """Results of query analysis"""
    query_type: QueryType
    complexity_score: float
    entities: List[str]
    intent: str
    suggested_strategy: RetrievalStrategy

class AdvancedChunker:
    """Advanced document chunking with multiple strategies"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.strategies = {
            'recursive': RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200),
            'sentence': SentenceTransformersTokenTextSplitter(chunk_overlap=50, tokens_per_chunk=256),
            'semantic': self._semantic_chunker,
            'hierarchical': self._hierarchical_chunker
        }
    
    def _semantic_chunker(self, text: str) -> List[str]:
        """Semantic-based chunking using sentence similarity"""
        doc = self.nlp(text)
        sentences = [sent.text for sent in doc.sents]
        
        if len(sentences) <= 1:
            return sentences
        
        # Use sentence transformer for semantic similarity
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(sentences)
        
        # Find optimal chunk boundaries based on similarity
        chunks = []
        current_chunk = [sentences[0]]
        
        for i in range(1, len(sentences)):
            similarity = cosine_similarity(
                embeddings[i-1].reshape(1, -1),
                embeddings[i].reshape(1, -1)
            )[0][0]
            
            if similarity > 0.7:  # High similarity threshold
                current_chunk.append(sentences[i])
            else:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentences[i]]
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _hierarchical_chunker(self, text: str) -> List[str]:
        """Hierarchical chunking with multiple levels"""
        # Level 1: Paragraph splitting
        paragraphs = text.split('\n\n')
        
        # Level 2: Sentence splitting within paragraphs
        chunks = []
        for para in paragraphs:
            if len(para) > 500:  # Large paragraphs
                doc = self.nlp(para)
                sentences = [sent.text for sent in doc.sents]
                
                # Group sentences into smaller chunks
                current_chunk = []
                current_length = 0
                
                for sent in sentences:
                    if current_length + len(sent) > 400:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        current_chunk = [sent]
                        current_length = len(sent)
                    else:
                        current_chunk.append(sent)
                        current_length += len(sent)
                
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
            else:
                chunks.append(para)
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def chunk_document(self, text: str, strategy: str = 'hybrid') -> List[Document]:
        """Chunk document using specified strategy"""
        if strategy == 'hybrid':
            # Combine multiple strategies
            recursive_chunks = self.strategies['recursive'].split_text(text)
            semantic_chunks = self._semantic_chunker(text)
            
            # Merge and deduplicate
            all_chunks = recursive_chunks + semantic_chunks
            unique_chunks = list(set(all_chunks))
            
            return [Document(page_content=chunk, metadata={'strategy': 'hybrid'}) 
                   for chunk in unique_chunks if len(chunk.strip()) > 50]
        else:
            strategy_func = self.strategies.get(strategy, self.strategies['recursive'])
            if callable(strategy_func):
                chunks = strategy_func(text)
            else:
                chunks = strategy_func.split_text(text)
            
            return [Document(page_content=chunk, metadata={'strategy': strategy}) 
                   for chunk in chunks if len(chunk.strip()) > 50]

class MultiModalProcessor:
    """Advanced document processor for multiple content types"""
    
    def __init__(self):
        self.image_caption_model = pipeline("image-to-text", 
                                           model="nlpconnect/vit-gpt2-image-captioning")
        self.table_extraction_tools = ['camelot', 'tabula', 'pytesseract']
    
    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Advanced PDF extraction with multi-modal content"""
        result = {
            'text': '',
            'tables': [],
            'images': [],
            'metadata': {}
        }
        
        try:
            reader = PdfReader(pdf_path)
            result['metadata'] = {
                'num_pages': len(reader.pages),
                'title': reader.metadata.get('/Title', ''),
                'author': reader.metadata.get('/Author', ''),
                'creation_date': reader.metadata.get('/CreationDate', '')
            }
            
            all_text = []
            
            for page_num, page in enumerate(reader.pages):
                # Extract text
                text = page.extract_text()
                if text:
                    all_text.append(text)
                
                # Extract images (if any)
                if hasattr(page, 'images'):
                    for img_idx, img in enumerate(page.images):
                        try:
                            # Convert image to caption
                            caption = self._process_image(img)
                            result['images'].append({
                                'page': page_num,
                                'index': img_idx,
                                'caption': caption
                            })
                        except Exception as e:
                            logger.warning(f"Failed to process image on page {page_num}: {e}")
            
            result['text'] = '\n\n'.join(all_text)
            
            # Extract tables
            result['tables'] = self._extract_tables(pdf_path)
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise
        
        return result
    
    def _process_image(self, image_data) -> str:
        """Generate caption for image"""
        try:
            # Convert image data to PIL Image
            # This is a simplified version - actual implementation would depend on image format
            caption = "Document image content"  # Placeholder
            return caption
        except Exception as e:
            logger.warning(f"Image processing failed: {e}")
            return "Image content not accessible"
    
    def _extract_tables(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF using multiple methods"""
        tables = []
        
        try:
            # Method 1: Camelot
            camelot_tables = camelot.read_pdf(pdf_path, pages='all')
            for i, table in enumerate(camelot_tables):
                tables.append({
                    'method': 'camelot',
                    'table_id': i,
                    'data': table.df.to_dict(),
                    'accuracy': getattr(table, 'accuracy', 0)
                })
        except Exception as e:
            logger.warning(f"Camelot extraction failed: {e}")
        
        try:
            # Method 2: Tabula
            tabula_tables = tabula.read_pdf(pdf_path, pages='all', multiple_tables=True)
            for i, table in enumerate(tabula_tables):
                tables.append({
                    'method': 'tabula',
                    'table_id': i,
                    'data': table.to_dict()
                })
        except Exception as e:
            logger.warning(f"Tabula extraction failed: {e}")
        
        return tables

class QueryClassifier:
    """Intelligent query classification and routing"""
    
    def __init__(self):
        self.classifier = pipeline("text-classification", 
                                 model="facebook/bart-large-mnli")
        self.ner_model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
        
        # Query type templates
        self.query_templates = {
            QueryType.FACTUAL: ["What is", "Who is", "When did", "Where is"],
            QueryType.ANALYTICAL: ["Why", "How", "Analyze", "Explain"],
            QueryType.COMPARISON: ["Compare", "Difference", "vs", "versus"],
            QueryType.TEMPORAL: ["trend", "over time", "year", "quarter"],
            QueryType.NUMERICAL: ["revenue", "profit", "cost", "amount", "percentage"],
            QueryType.CONCEPTUAL: ["concept", "idea", "theory", "principle"]
        }
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Comprehensive query analysis"""
        # Extract entities
        entities = self._extract_entities(query)
        
        # Classify query type
        query_type = self._classify_query_type(query)
        
        # Calculate complexity
        complexity = self._calculate_complexity(query)
        
        # Determine intent
        intent = self._determine_intent(query)
        
        # Suggest retrieval strategy
        strategy = self._suggest_strategy(query_type, complexity)
        
        return QueryAnalysis(
            query_type=query_type,
            complexity_score=complexity,
            entities=entities,
            intent=intent,
            suggested_strategy=strategy
        )
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract named entities from query"""
        entities = self.ner_model(query)
        return [entity['word'] for entity in entities if entity['score'] > 0.8]
    
    def _classify_query_type(self, query: str) -> QueryType:
        """Classify the type of query"""
        query_lower = query.lower()
        
        for query_type, templates in self.query_templates.items():
            if any(template.lower() in query_lower for template in templates):
                return query_type
        
        return QueryType.FACTUAL  # Default
    
    def _calculate_complexity(self, query: str) -> float:
        """Calculate query complexity score"""
        factors = {
            'length': len(query.split()) / 20,  # Normalized by average length
            'entities': len(self._extract_entities(query)) / 5,
            'question_words': len([w for w in query.lower().split() 
                                 if w in ['what', 'when', 'where', 'why', 'how', 'who']]) / 3,
            'conjunctions': len([w for w in query.lower().split() 
                               if w in ['and', 'or', 'but', 'however', 'although']]) / 2
        }
        
        return min(sum(factors.values()), 1.0)
    
    def _determine_intent(self, query: str) -> str:
        """Determine the user's intent"""
        intent_keywords = {
            'information_seeking': ['what', 'who', 'when', 'where'],
            'analysis': ['why', 'how', 'analyze', 'explain'],
            'comparison': ['compare', 'difference', 'better', 'worse'],
            'calculation': ['calculate', 'compute', 'total', 'sum']
        }
        
        query_lower = query.lower()
        for intent, keywords in intent_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return intent
        
        return 'general_inquiry'
    
    def _suggest_strategy(self, query_type: QueryType, complexity: float) -> RetrievalStrategy:
        """Suggest optimal retrieval strategy based on query analysis"""
        if complexity > 0.7:
            return RetrievalStrategy.HYBRID
        elif query_type in [QueryType.FACTUAL, QueryType.NUMERICAL]:
            return RetrievalStrategy.DENSE_ONLY
        elif query_type in [QueryType.ANALYTICAL, QueryType.CONCEPTUAL]:
            return RetrievalStrategy.MULTI_VECTOR
        else:
            return RetrievalStrategy.HYBRID

class HybridRetriever:
    """Advanced hybrid retrieval combining multiple approaches"""
    
    def __init__(self, collection_name: str = "advanced_rag"):
        self.collection_name = collection_name
        
        # Initialize models
        self.dense_model = SentenceTransformer('all-mpnet-base-v2')
        self.sparse_vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # DPR models
        self.dpr_question_encoder = DPRQuestionEncoder.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        self.dpr_context_encoder = DPRContextEncoder.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        self.dpr_question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )
        self.dpr_context_tokenizer = DPRContextEncoderTokenizer.from_pretrained(
            "facebook/dpr-ctx_encoder-single-nq-base"
        )
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client()
        self._setup_collections()
        
        # Document storage
        self.documents = []
        self.sparse_matrix = None
    
    def _setup_collections(self):
        """Setup ChromaDB collections for different embedding types"""
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except:
            pass
        
        self.dense_collection = self.chroma_client.create_collection(
            f"{self.collection_name}_dense"
        )
        
        self.dpr_collection = self.chroma_client.create_collection(
            f"{self.collection_name}_dpr"
        )
    
    def add_documents(self, documents: List[Document]):
        """Add documents to all retrieval indexes"""
        texts = [doc.page_content for doc in documents]
        metadata = [doc.metadata for doc in documents]
        
        # Store documents
        self.documents = documents
        
        # Dense embeddings
        dense_embeddings = self.dense_model.encode(texts).tolist()
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        self.dense_collection.add(
            ids=ids,
            documents=texts,
            embeddings=dense_embeddings,
            metadatas=metadata
        )
        
        # DPR embeddings
        dpr_embeddings = []
        for text in texts:
            inputs = self.dpr_context_tokenizer(text, return_tensors="pt", truncation=True)
            embedding = self.dpr_context_encoder(**inputs).pooler_output.detach().numpy()[0]
            dpr_embeddings.append(embedding.tolist())
        
        self.dpr_collection.add(
            ids=ids,
            documents=texts,
            embeddings=dpr_embeddings,
            metadatas=metadata
        )
        
        # Sparse matrix (TF-IDF)
        self.sparse_matrix = self.sparse_vectorizer.fit_transform(texts)
        
        logger.info(f"Added {len(documents)} documents to hybrid retriever")
    
    def retrieve(self, query: str, strategy: RetrievalStrategy, 
                k: int = 10) -> RetrievalResult:
        """Retrieve documents using specified strategy"""
        start_time = datetime.now()
        
        if strategy == RetrievalStrategy.DENSE_ONLY:
            result = self._dense_retrieve(query, k)
        elif strategy == RetrievalStrategy.SPARSE_ONLY:
            result = self._sparse_retrieve(query, k)
        elif strategy == RetrievalStrategy.DPR:
            result = self._dpr_retrieve(query, k)
        elif strategy == RetrievalStrategy.HYBRID:
            result = self._hybrid_retrieve(query, k)
        elif strategy == RetrievalStrategy.MULTI_VECTOR:
            result = self._multi_vector_retrieve(query, k)
        else:
            result = self._hybrid_retrieve(query, k)  # Default
        
        result.retrieval_time = (datetime.now() - start_time).total_seconds()
        result.strategy_used = strategy.value
        
        return result
    
    def _dense_retrieve(self, query: str, k: int) -> RetrievalResult:
        """Dense retrieval using sentence transformers"""
        query_embedding = self.dense_model.encode([query])
        
        results = self.dense_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=k
        )
        
        return RetrievalResult(
            documents=results['documents'][0],
            scores=results['distances'][0],
            metadata=results['metadatas'][0],
            strategy_used="dense",
            query_embedding=query_embedding[0]
        )
    
    def _sparse_retrieve(self, query: str, k: int) -> RetrievalResult:
        """Sparse retrieval using TF-IDF"""
        if self.sparse_matrix is None:
            raise ValueError("Sparse matrix not initialized. Add documents first.")
        
        query_vector = self.sparse_vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.sparse_matrix).flatten()
        
        top_indices = np.argsort(similarities)[::-1][:k]
        
        return RetrievalResult(
            documents=[self.documents[i].page_content for i in top_indices],
            scores=similarities[top_indices].tolist(),
            metadata=[self.documents[i].metadata for i in top_indices],
            strategy_used="sparse"
        )
    
    def _dpr_retrieve(self, query: str, k: int) -> RetrievalResult:
        """DPR-based retrieval"""
        question_inputs = self.dpr_question_tokenizer(query, return_tensors="pt")
        question_embedding = self.dpr_question_encoder(**question_inputs).pooler_output
        
        results = self.dpr_collection.query(
            query_embeddings=question_embedding.detach().numpy().tolist(),
            n_results=k
        )
        
        return RetrievalResult(
            documents=results['documents'][0],
            scores=results['distances'][0],
            metadata=results['metadatas'][0],
            strategy_used="dpr",
            query_embedding=question_embedding.detach().numpy()[0]
        )
    
    def _hybrid_retrieve(self, query: str, k: int) -> RetrievalResult:
        """Hybrid retrieval combining dense and sparse"""
        # Get results from both methods
        dense_result = self._dense_retrieve(query, k)
        sparse_result = self._sparse_retrieve(query, k)
        
        # Combine and re-rank
        all_docs = dense_result.documents + sparse_result.documents
        all_scores = dense_result.scores + sparse_result.scores
        all_metadata = dense_result.metadata + sparse_result.metadata
        
        # Remove duplicates while preserving order
        seen = set()
        unique_docs = []
        unique_scores = []
        unique_metadata = []
        
        for doc, score, meta in zip(all_docs, all_scores, all_metadata):
            if doc not in seen:
                seen.add(doc)
                unique_docs.append(doc)
                unique_scores.append(score)
                unique_metadata.append(meta)
        
        # Re-rank using cross-encoder
        if len(unique_docs) > 1:
            pairs = [(query, doc) for doc in unique_docs]
            cross_scores = self.cross_encoder.predict(pairs)
            
            # Sort by cross-encoder scores
            sorted_indices = np.argsort(cross_scores)[::-1]
            
            unique_docs = [unique_docs[i] for i in sorted_indices]
            unique_scores = [cross_scores[i] for i in sorted_indices]
            unique_metadata = [unique_metadata[i] for i in sorted_indices]
        
        return RetrievalResult(
            documents=unique_docs[:k],
            scores=unique_scores[:k],
            metadata=unique_metadata[:k],
            strategy_used="hybrid",
            query_embedding=dense_result.query_embedding
        )
    
    def _multi_vector_retrieve(self, query: str, k: int) -> RetrievalResult:
        """Multi-vector retrieval using ensemble of models"""
        # Get results from all methods
        dense_result = self._dense_retrieve(query, k * 2)
        dpr_result = self._dpr_retrieve(query, k * 2)
        sparse_result = self._sparse_retrieve(query, k * 2)
        
        # Weighted combination
        weights = {'dense': 0.4, 'dpr': 0.4, 'sparse': 0.2}
        
        # Score normalization and combination
        doc_scores = {}
        
        for result, weight in zip([dense_result, dpr_result, sparse_result], 
                                [weights['dense'], weights['dpr'], weights['sparse']]):
            normalized_scores = np.array(result.scores)
            if len(normalized_scores) > 0:
                normalized_scores = (normalized_scores - normalized_scores.min()) / (normalized_scores.max() - normalized_scores.min() + 1e-8)
            
            for doc, score, meta in zip(result.documents, normalized_scores, result.metadata):
                if doc not in doc_scores:
                    doc_scores[doc] = {'score': 0, 'metadata': meta}
                doc_scores[doc]['score'] += score * weight
        
        # Sort by combined scores
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1]['score'], reverse=True)
        
        final_docs = [item[0] for item in sorted_docs[:k]]
        final_scores = [item[1]['score'] for item in sorted_docs[:k]]
        final_metadata = [item[1]['metadata'] for item in sorted_docs[:k]]
        
        return RetrievalResult(
            documents=final_docs,
            scores=final_scores,
            metadata=final_metadata,
            strategy_used="multi_vector",
            query_embedding=dense_result.query_embedding
        )

class MemoryManager:
    """Advanced memory management for conversation history and learning"""
    
    def __init__(self, max_history: int = 100):
        self.conversation_history = []
        self.user_feedback = []
        self.query_patterns = {}
        self.max_history = max_history
        self.memory_file = "rag_memory.pkl"
        self.load_memory()
    
    def add_interaction(self, query: str, response: str, feedback: Optional[float] = None):
        """Add interaction to memory"""
        interaction = {
            'timestamp': datetime.now(),
            'query': query,
            'response': response,
            'feedback': feedback,
            'query_id': str(uuid.uuid4())
        }
        
        self.conversation_history.append(interaction)
        
        # Maintain max history
        if len(self.conversation_history) > self.max_history:
            self.conversation_history = self.conversation_history[-self.max_history:]
        
        # Update query patterns
        self._update_patterns(query, feedback)
        
        self.save_memory()
    
    def _update_patterns(self, query: str, feedback: Optional[float]):
        """Update query patterns for learning"""
        query_type = self._categorize_query(query)
        
        if query_type not in self.query_patterns:
            self.query_patterns[query_type] = {
                'count': 0,
                'avg_feedback': 0,
                'successful_strategies': []
            }
        
        self.query_patterns[query_type]['count'] += 1
        
        if feedback is not None:
            current_avg = self.query_patterns[query_type]['avg_feedback']
            current_count = self.query_patterns[query_type]['count']
            
            # Update running average
            new_avg = (current_avg * (current_count - 1) + feedback) / current_count
            self.query_patterns[query_type]['avg_feedback'] = new_avg
    
    def _categorize_query(self, query: str) -> str:
        """Simple query categorization"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['revenue', 'profit', 'financial']):
            return 'financial'
        elif any(word in query_lower for word in ['compare', 'vs', 'difference']):
            return 'comparison'
        elif any(word in query_lower for word in ['trend', 'over time', 'growth']):
            return 'temporal'
        else:
            return 'general'
    
    def get_relevant_history(self, query: str, n: int = 3) -> List[Dict]:
        """Get relevant conversation history"""
        if not self.conversation_history:
            return []
        
        # Simple similarity-based retrieval
        model = SentenceTransformer('all-MiniLM-L6-v2')
        query_embedding = model.encode([query])
        
        history_queries = [item['query'] for item in self.conversation_history]
        history_embeddings = model.encode(history_queries)
        
        similarities = cosine_similarity(query_embedding, history_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:n]
        
        return [self.conversation_history[i] for i in top_indices if similarities[i] > 0.5]
    
    def save_memory(self):
        """Save memory to disk"""
        try:
            with open(self.memory_file, 'wb') as f:
                pickle.dump({
                    'conversation_history': self.conversation_history,
                    'user_feedback': self.user_feedback,
                    'query_patterns': self.query_patterns
                }, f)
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
    
    def load_memory(self):
        """Load memory from disk"""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    data = pickle.load(f)
                    self.conversation_history = data.get('conversation_history', [])
                    self.user_feedback = data.get('user_feedback', [])
                    self.query_patterns = data.get('query_patterns', {})
        except Exception as e:
            logger.error(f"Failed to load memory: {e}") 