"""
Advanced RAG System - Part 2
============================

This module contains:
- RAG Evaluation and Metrics
- API Interface with FastAPI
- Advanced Visualization
- Main RAG Orchestrator
- Production Features
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Evaluation Libraries
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_relevancy,
        context_precision,
        context_recall,
        faithfulness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    print("Warning: RAGAS not available. Install with: pip install ragas datasets")
    RAGAS_AVAILABLE = False

# API Libraries
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel, Field

# Import from part 1
from advanced_rag_system import (
    QueryType, RetrievalStrategy, RetrievalResult, QueryAnalysis,
    AdvancedChunker, MultiModalProcessor, QueryClassifier,
    HybridRetriever, MemoryManager
)

class RAGEvaluator:
    """Comprehensive RAG evaluation system"""
    
    def __init__(self):
        if RAGAS_AVAILABLE:
            self.metrics = {
                'answer_relevancy': answer_relevancy,
                'context_relevancy': context_relevancy,  
                'context_precision': context_precision,
                'context_recall': context_recall,
                'faithfulness': faithfulness
            }
        else:
            self.metrics = {}
        
        self.evaluation_history = []
    
    def evaluate_retrieval(self, queries: List[str], retrieved_docs: List[List[str]], 
                          ground_truth: List[List[str]]) -> Dict[str, float]:
        """Evaluate retrieval performance"""
        metrics = {}
        
        # Precision@k
        precisions = []
        for retrieved, truth in zip(retrieved_docs, ground_truth):
            if truth:
                precision = len(set(retrieved) & set(truth)) / len(retrieved)
                precisions.append(precision)
        metrics['precision_at_k'] = np.mean(precisions) if precisions else 0
        
        # Recall@k
        recalls = []
        for retrieved, truth in zip(retrieved_docs, ground_truth):
            if truth:
                recall = len(set(retrieved) & set(truth)) / len(truth)
                recalls.append(recall)
        metrics['recall_at_k'] = np.mean(recalls) if recalls else 0
        
        # F1 Score
        if metrics['precision_at_k'] + metrics['recall_at_k'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision_at_k'] * metrics['recall_at_k']) / \
                                 (metrics['precision_at_k'] + metrics['recall_at_k'])
        else:
            metrics['f1_score'] = 0
        
        # Mean Reciprocal Rank (MRR)
        mrr_scores = []
        for retrieved, truth in zip(retrieved_docs, ground_truth):
            for i, doc in enumerate(retrieved):
                if doc in truth:
                    mrr_scores.append(1 / (i + 1))
                    break
            else:
                mrr_scores.append(0)
        metrics['mrr'] = np.mean(mrr_scores)
        
        return metrics
    
    def evaluate_generation(self, questions: List[str], answers: List[str], 
                           contexts: List[List[str]], ground_truth: List[str]) -> Dict[str, float]:
        """Evaluate generation quality using RAGAS"""
        if not RAGAS_AVAILABLE:
            print("RAGAS not available. Skipping generation evaluation.")
            return {}
            
        try:
            # Prepare dataset for RAGAS
            data = {
                'question': questions,
                'answer': answers,
                'contexts': contexts,
                'ground_truths': ground_truth
            }
            dataset = Dataset.from_dict(data)
            
            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=list(self.metrics.values())
            )
            
            return dict(result)
        
        except Exception as e:
            logger.error(f"Generation evaluation failed: {e}")
            return {}
    
    def benchmark_strategies(self, queries: List[str], 
                           retriever: 'HybridRetriever') -> Dict[str, Dict[str, float]]:
        """Benchmark different retrieval strategies"""
        strategies = [
            RetrievalStrategy.DENSE_ONLY,
            RetrievalStrategy.SPARSE_ONLY,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.DPR
        ]
        
        results = {}
        
        for strategy in strategies:
            start_time = time.time()
            strategy_results = []
            
            for query in queries:
                try:
                    result = retriever.retrieve(query, strategy, k=5)
                    strategy_results.append(result)
                except Exception as e:
                    logger.error(f"Strategy {strategy} failed for query '{query}': {e}")
                    continue
            
            end_time = time.time()
            
            # Calculate metrics
            avg_retrieval_time = (end_time - start_time) / len(queries)
            avg_score = np.mean([np.mean(r.scores) for r in strategy_results if r.scores])
            
            results[strategy.value] = {
                'avg_retrieval_time': avg_retrieval_time,
                'avg_score': avg_score,
                'success_rate': len(strategy_results) / len(queries)
            }
        
        return results

class GuardrailSystem:
    """Safety and quality guardrails for RAG system"""
    
    def __init__(self):
        self.toxic_classifier = pipeline("text-classification", 
                                        model="unitary/toxic-bert")
        self.hallucination_detector = pipeline("text-classification",
                                              model="microsoft/DialoGPT-medium")
        
        self.safety_thresholds = {
            'toxicity': 0.1,
            'confidence': 0.7,
            'relevance': 0.5
        }
    
    def check_query_safety(self, query: str) -> Dict[str, Any]:
        """Check if query is safe and appropriate"""
        results = {
            'is_safe': True,
            'issues': [],
            'confidence': 1.0
        }
        
        try:
            # Toxicity check
            toxic_result = self.toxic_classifier(query)[0]
            if toxic_result['label'] == 'TOXIC' and toxic_result['score'] > self.safety_thresholds['toxicity']:
                results['is_safe'] = False
                results['issues'].append('toxic_content')
                results['confidence'] = toxic_result['score']
        
        except Exception as e:
            logger.warning(f"Safety check failed: {e}")
            results['issues'].append('safety_check_failed')
        
        return results
    
    def check_response_quality(self, query: str, response: str, 
                             retrieved_docs: List[str]) -> Dict[str, Any]:
        """Check response quality and detect potential hallucinations"""
        results = {
            'quality_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        try:
            # Check if response is grounded in retrieved documents
            response_lower = response.lower()
            doc_overlap = 0
            
            for doc in retrieved_docs:
                doc_words = set(doc.lower().split())
                response_words = set(response_lower.split())
                overlap = len(doc_words & response_words) / len(response_words)
                doc_overlap = max(doc_overlap, overlap)
            
            if doc_overlap < self.safety_thresholds['relevance']:
                results['issues'].append('low_grounding')
                results['recommendations'].append('increase_retrieval_scope')
            
            results['quality_score'] = doc_overlap
            
        except Exception as e:
            logger.warning(f"Quality check failed: {e}")
            results['issues'].append('quality_check_failed')
        
        return results

class AdvancedVisualizer:
    """Advanced visualization for RAG system analytics"""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def plot_embedding_space(self, embeddings: np.ndarray, labels: List[str],
                           query_embedding: Optional[np.ndarray] = None,
                           title: str = "Embedding Space Visualization") -> go.Figure:
        """Interactive 3D visualization of embedding space"""
        
        # UMAP for dimensionality reduction
        umap_3d = umap.UMAP(n_components=3, random_state=42)
        reduced_embeddings = umap_3d.fit_transform(embeddings)
        
        fig = go.Figure()
        
        # Plot document embeddings
        fig.add_trace(go.Scatter3d(
            x=reduced_embeddings[:, 0],
            y=reduced_embeddings[:, 1], 
            z=reduced_embeddings[:, 2],
            mode='markers',
            marker=dict(size=5, color=self.color_palette[0], opacity=0.6),
            text=labels,
            name='Documents',
            hovertemplate='<b>%{text}</b><br>' +
                         'X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
        ))
        
        # Plot query embedding if provided
        if query_embedding is not None:
            query_reduced = umap_3d.transform(query_embedding.reshape(1, -1))
            fig.add_trace(go.Scatter3d(
                x=query_reduced[:, 0],
                y=query_reduced[:, 1],
                z=query_reduced[:, 2],
                mode='markers',
                marker=dict(size=15, color='red', symbol='diamond'),
                name='Query',
                hovertemplate='<b>Query</b><br>' +
                             'X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>'
            ))
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="UMAP 1",
                yaxis_title="UMAP 2", 
                zaxis_title="UMAP 3"
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def plot_retrieval_performance(self, benchmark_results: Dict[str, Dict[str, float]]) -> go.Figure:
        """Plot retrieval strategy performance comparison"""
        
        strategies = list(benchmark_results.keys())
        metrics = ['avg_retrieval_time', 'avg_score', 'success_rate']
        
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Retrieval Time (s)', 'Average Score', 'Success Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        for i, metric in enumerate(metrics):
            values = [benchmark_results[strategy][metric] for strategy in strategies]
            
            fig.add_trace(
                go.Bar(
                    x=strategies,
                    y=values,
                    name=metric,
                    marker_color=self.color_palette[i]
                ),
                row=1, col=i+1
            )
        
        fig.update_layout(
            title="Retrieval Strategy Performance Comparison",
            showlegend=False,
            height=400
        )
        
        return fig
    
    def plot_query_analytics(self, memory_manager: MemoryManager) -> go.Figure:
        """Plot query analytics and patterns"""
        
        if not memory_manager.query_patterns:
            return go.Figure().add_annotation(text="No query data available")
        
        patterns = memory_manager.query_patterns
        categories = list(patterns.keys())
        counts = [patterns[cat]['count'] for cat in categories]
        avg_feedback = [patterns[cat]['avg_feedback'] for cat in categories]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Query Distribution', 'Average Feedback by Category'),
            specs=[[{"type": "domain"}, {"secondary_y": False}]]
        )
        
        # Pie chart for query distribution
        fig.add_trace(
            go.Pie(
                labels=categories,
                values=counts,
                name="Query Distribution"
            ),
            row=1, col=1
        )
        
        # Bar chart for feedback
        fig.add_trace(
            go.Bar(
                x=categories,
                y=avg_feedback,
                name="Average Feedback",
                marker_color=self.color_palette[1]
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title="Query Analytics Dashboard",
            height=400
        )
        
        return fig

# API Models
class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's query")
    strategy: Optional[str] = Field(None, description="Retrieval strategy to use")
    k: int = Field(5, description="Number of documents to retrieve")
    include_analysis: bool = Field(True, description="Include query analysis")

class QueryResponse(BaseModel):
    answer: str
    retrieved_documents: List[str]
    sources: List[Dict[str, Any]]
    query_analysis: Optional[Dict[str, Any]]
    retrieval_strategy: str
    confidence_score: float
    processing_time: float

class FeedbackRequest(BaseModel):
    query_id: str
    rating: float = Field(..., ge=0, le=5)
    comments: Optional[str] = None

# FastAPI Application
app = FastAPI(
    title="Advanced RAG System API",
    description="Production-ready RAG system with multiple retrieval strategies",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global system instance
rag_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    global rag_system
    rag_system = AdvancedRAGSystem()
    logger.info("Advanced RAG System initialized")

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest) -> QueryResponse:
    """Main query endpoint"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        result = await rag_system.process_query_async(
            query=request.query,
            strategy=request.strategy,
            k=request.k,
            include_analysis=request.include_analysis
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Query processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """Submit user feedback"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    rag_system.add_feedback(request.query_id, request.rating, request.comments)
    return {"message": "Feedback recorded successfully"}

@app.get("/analytics")
async def analytics_endpoint():
    """Get system analytics"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    return rag_system.get_analytics()

@app.post("/upload_document")
async def upload_document_endpoint(file_path: str):
    """Upload and process a new document"""
    if not rag_system:
        raise HTTPException(status_code=500, detail="RAG system not initialized")
    
    try:
        await rag_system.add_document_async(file_path)
        return {"message": "Document processed successfully"}
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class AdvancedRAGSystem:
    """Main orchestrator for the advanced RAG system"""
    
    def __init__(self, collection_name: str = "advanced_rag"):
        logger.info("Initializing Advanced RAG System...")
        
        # Initialize components
        self.chunker = AdvancedChunker()
        self.processor = MultiModalProcessor()
        self.classifier = QueryClassifier()
        self.retriever = HybridRetriever(collection_name)
        self.memory = MemoryManager()
        self.evaluator = RAGEvaluator()
        self.guardrails = GuardrailSystem()
        self.visualizer = AdvancedVisualizer()
        
        # OpenAI setup
        openai.api_key = os.getenv("OPENAI_API_KEY")
        
        # Performance tracking
        self.query_count = 0
        self.performance_metrics = {
            'total_queries': 0,
            'avg_response_time': 0,
            'success_rate': 0
        }
        
        logger.info("Advanced RAG System initialized successfully")
    
    async def add_document_async(self, file_path: str, chunking_strategy: str = "hybrid"):
        """Asynchronously add document to the system"""
        logger.info(f"Processing document: {file_path}")
        
        # Multi-modal processing
        extracted_content = self.processor.extract_text_from_pdf(file_path)
        
        # Advanced chunking
        documents = self.chunker.chunk_document(
            extracted_content['text'], 
            strategy=chunking_strategy
        )
        
        # Add metadata
        for doc in documents:
            doc.metadata.update({
                'source': file_path,
                'processed_at': datetime.now().isoformat(),
                'num_tables': len(extracted_content.get('tables', [])),
                'num_images': len(extracted_content.get('images', []))
            })
        
        # Add to retriever
        self.retriever.add_documents(documents)
        
        logger.info(f"Successfully processed {len(documents)} chunks from {file_path}")
        
        return {
            'chunks_created': len(documents),
            'tables_extracted': len(extracted_content.get('tables', [])),
            'images_processed': len(extracted_content.get('images', []))
        }
    
    async def process_query_async(self, query: str, strategy: Optional[str] = None,
                                k: int = 5, include_analysis: bool = True) -> QueryResponse:
        """Asynchronously process a user query"""
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            # Safety check
            safety_check = self.guardrails.check_query_safety(query)
            if not safety_check['is_safe']:
                raise HTTPException(status_code=400, detail="Query violates safety guidelines")
            
            # Query analysis
            analysis = None
            if include_analysis:
                analysis = self.classifier.analyze_query(query)
                suggested_strategy = analysis.suggested_strategy
            else:
                suggested_strategy = RetrievalStrategy.HYBRID
            
            # Use provided strategy or suggested one
            if strategy:
                try:
                    retrieval_strategy = RetrievalStrategy(strategy)
                except ValueError:
                    retrieval_strategy = suggested_strategy
            else:
                retrieval_strategy = suggested_strategy
            
            # Retrieve relevant documents
            retrieval_result = self.retriever.retrieve(query, retrieval_strategy, k)
            
            # Generate response
            response = await self._generate_response_async(query, retrieval_result)
            
            # Quality check
            quality_check = self.guardrails.check_response_quality(
                query, response, retrieval_result.documents
            )
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence(retrieval_result, quality_check)
            
            processing_time = time.time() - start_time
            
            # Update memory
            self.memory.add_interaction(query, response)
            
            # Update performance metrics
            self._update_performance_metrics(processing_time, True)
            
            # Prepare response
            query_response = QueryResponse(
                answer=response,
                retrieved_documents=retrieval_result.documents,
                sources=[{'content': doc, 'metadata': meta} 
                        for doc, meta in zip(retrieval_result.documents, 
                                           retrieval_result.metadata)],
                query_analysis=analysis.__dict__ if analysis else None,
                retrieval_strategy=retrieval_strategy.value,
                confidence_score=confidence_score,
                processing_time=processing_time
            )
            
            return query_response
            
        except Exception as e:
            self._update_performance_metrics(time.time() - start_time, False)
            logger.error(f"Query processing failed: {e}")
            raise
    
    async def _generate_response_async(self, query: str, 
                                     retrieval_result: RetrievalResult) -> str:
        """Generate response using OpenAI with retrieved context"""
        
        # Get relevant conversation history
        relevant_history = self.memory.get_relevant_history(query, n=2)
        
        # Prepare context
        context = "\n\n".join(retrieval_result.documents)
        
        # Build prompt with history
        history_context = ""
        if relevant_history:
            history_context = "\n\nRelevant conversation history:\n"
            for item in relevant_history:
                history_context += f"Q: {item['query']}\nA: {item['response']}\n"
        
        prompt = f"""
        You are an expert assistant analyzing documents. Use the provided context to answer the user's question accurately and comprehensively.
        
        Context:
        {context}
        {history_context}
        
        Question: {query}
        
        Instructions:
        1. Base your answer strictly on the provided context
        2. If the context doesn't contain enough information, say so explicitly
        3. Cite specific parts of the context when possible
        4. Be concise but comprehensive
        5. Use a professional but friendly tone
        
        Answer:
        """
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.1
            )
            
            return response['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            logger.error(f"Response generation failed: {e}")
            return "I apologize, but I'm unable to generate a response at this time. Please try again."
    
    def _calculate_confidence(self, retrieval_result: RetrievalResult, 
                            quality_check: Dict[str, Any]) -> float:
        """Calculate confidence score for the response"""
        factors = {
            'retrieval_scores': np.mean(retrieval_result.scores) if retrieval_result.scores else 0,
            'quality_score': quality_check.get('quality_score', 0),
            'num_documents': min(len(retrieval_result.documents) / 5, 1),  # Normalized
            'strategy_reliability': 0.8 if retrieval_result.strategy_used == 'hybrid' else 0.6
        }
        
        # Weighted average
        weights = {'retrieval_scores': 0.3, 'quality_score': 0.4, 
                  'num_documents': 0.2, 'strategy_reliability': 0.1}
        
        confidence = sum(factors[key] * weights[key] for key in factors)
        return min(max(confidence, 0), 1)  # Clamp between 0 and 1
    
    def _update_performance_metrics(self, processing_time: float, success: bool):
        """Update system performance metrics"""
        self.performance_metrics['total_queries'] += 1
        
        # Update average response time
        current_avg = self.performance_metrics['avg_response_time']
        total_queries = self.performance_metrics['total_queries']
        
        new_avg = (current_avg * (total_queries - 1) + processing_time) / total_queries
        self.performance_metrics['avg_response_time'] = new_avg
        
        # Update success rate
        if success:
            self.query_count += 1
        
        self.performance_metrics['success_rate'] = self.query_count / total_queries
    
    def add_feedback(self, query_id: str, rating: float, comments: Optional[str] = None):
        """Add user feedback to the system"""
        feedback = {
            'query_id': query_id,
            'rating': rating,
            'comments': comments,
            'timestamp': datetime.now()
        }
        
        self.memory.user_feedback.append(feedback)
        self.memory.save_memory()
        
        logger.info(f"Feedback recorded for query {query_id}: {rating}/5")
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive system analytics"""
        return {
            'performance_metrics': self.performance_metrics,
            'query_patterns': self.memory.query_patterns,
            'total_documents': len(self.retriever.documents),
            'memory_size': len(self.memory.conversation_history),
            'feedback_count': len(self.memory.user_feedback),
            'avg_feedback_rating': np.mean([f['rating'] for f in self.memory.user_feedback]) 
                                 if self.memory.user_feedback else 0
        }
    
    def benchmark_system(self, test_queries: List[str]) -> Dict[str, Any]:
        """Run comprehensive system benchmark"""
        logger.info("Running system benchmark...")
        
        # Benchmark retrieval strategies
        retrieval_benchmark = self.evaluator.benchmark_strategies(test_queries, self.retriever)
        
        # Test query processing
        processing_times = []
        success_count = 0
        
        for query in test_queries:
            try:
                start_time = time.time()
                asyncio.run(self.process_query_async(query, include_analysis=False))
                processing_times.append(time.time() - start_time)
                success_count += 1
            except Exception as e:
                logger.error(f"Benchmark query failed: {e}")
        
        return {
            'retrieval_benchmark': retrieval_benchmark,
            'avg_processing_time': np.mean(processing_times) if processing_times else 0,
            'success_rate': success_count / len(test_queries),
            'total_test_queries': len(test_queries)
        }

# Streamlit Interface
def create_streamlit_app():
    """Create Streamlit interface for the RAG system"""
    st.set_page_config(
        page_title="Advanced RAG System",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 Advanced RAG System")
    st.markdown("Intelligent document retrieval and question answering with multiple strategies")
    
    # Initialize session state
    if 'rag_system' not in st.session_state:
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = AdvancedRAGSystem()
    
    # Sidebar
    st.sidebar.title("⚙️ System Controls")
    
    # Document upload
    st.sidebar.subheader("📄 Document Management")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=['pdf'])
    
    if uploaded_file and st.sidebar.button("Process Document"):
        with st.spinner("Processing document..."):
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            
            try:
                result = asyncio.run(st.session_state.rag_system.add_document_async(temp_path))
                st.sidebar.success(f"✅ Processed {result['chunks_created']} chunks")
                os.remove(temp_path)
            except Exception as e:
                st.sidebar.error(f"❌ Error: {e}")
                if os.path.exists(temp_path):
                    os.remove(temp_path)
    
    # Strategy selection
    st.sidebar.subheader("🎯 Retrieval Strategy")
    strategy = st.sidebar.selectbox(
        "Select strategy:",
        ["auto", "dense_only", "sparse_only", "hybrid", "dpr", "multi_vector"]
    )
    
    k_docs = st.sidebar.slider("Documents to retrieve:", 1, 20, 5)
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Query Interface")
        
        # Query input
        query = st.text_area("Enter your question:", height=100)
        
        if st.button("🔍 Search", type="primary") and query:
            with st.spinner("Processing query..."):
                try:
                    strategy_param = None if strategy == "auto" else strategy
                    result = asyncio.run(st.session_state.rag_system.process_query_async(
                        query=query,
                        strategy=strategy_param,
                        k=k_docs
                    ))
                    
                    # Display results
                    st.subheader("📋 Answer")
                    st.write(result.answer)
                    
                    # Show metadata
                    col_meta1, col_meta2, col_meta3 = st.columns(3)
                    with col_meta1:
                        st.metric("Confidence", f"{result.confidence_score:.2f}")
                    with col_meta2:
                        st.metric("Processing Time", f"{result.processing_time:.2f}s")
                    with col_meta3:
                        st.metric("Strategy Used", result.retrieval_strategy)
                    
                    # Retrieved documents
                    st.subheader("📚 Retrieved Documents")
                    for i, doc in enumerate(result.retrieved_documents):
                        with st.expander(f"Document {i+1}"):
                            st.write(doc[:500] + "..." if len(doc) > 500 else doc)
                    
                    # Query analysis
                    if result.query_analysis:
                        st.subheader("🔍 Query Analysis")
                        analysis = result.query_analysis
                        st.json(analysis)
                    
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    with col2:
        st.subheader("📊 System Analytics")
        
        if st.button("🔄 Refresh Analytics"):
            analytics = st.session_state.rag_system.get_analytics()
            
            # Performance metrics
            st.metric("Total Queries", analytics['performance_metrics']['total_queries'])
            st.metric("Success Rate", f"{analytics['performance_metrics']['success_rate']:.2%}")
            st.metric("Avg Response Time", f"{analytics['performance_metrics']['avg_response_time']:.2f}s")
            
            # Query patterns visualization
            if analytics['query_patterns']:
                st.subheader("📈 Query Patterns")
                fig = st.session_state.rag_system.visualizer.plot_query_analytics(
                    st.session_state.rag_system.memory
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    # Run FastAPI server
    print("Starting Advanced RAG System...")
    print("FastAPI server will be available at: http://localhost:8000")
    print("Streamlit app: run 'streamlit run advanced_rag_system_part2.py'")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)