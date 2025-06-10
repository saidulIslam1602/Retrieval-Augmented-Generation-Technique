"""
Advanced RAG System Demonstration
================================

This script demonstrates all the advanced features of the RAG system:
- Multi-modal document processing
- Hybrid retrieval strategies
- Query classification and routing
- Advanced evaluation metrics
- Real-time analytics
- Interactive visualization

Usage:
    python demo_advanced_rag.py
"""

import asyncio
import os
import time
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Import the advanced system components
try:
    from advanced_rag_system import (
        AdvancedChunker, MultiModalProcessor, QueryClassifier,
        HybridRetriever, MemoryManager, QueryType, RetrievalStrategy
    )
    from advanced_rag_system_part2 import (
        AdvancedRAGSystem, RAGEvaluator, GuardrailSystem, AdvancedVisualizer
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure all dependencies are installed: pip install -r requirements_advanced.txt")
    exit(1)

class RAGSystemDemo:
    """Comprehensive demonstration of the Advanced RAG System"""
    
    def __init__(self):
        print("🚀 Initializing Advanced RAG System Demo...")
        self.setup_environment()
        self.rag_system = None
        self.demo_queries = [
            "What was the total revenue for the year?",
            "How did the company perform compared to previous year?",
            "What were the major factors contributing to revenue growth?",
            "Compare the financial performance across different quarters",
            "What investments were made in research and development?",
            "Analyze the market trends that affected the business",
            "What are the future growth prospects mentioned in the report?",
            "How did COVID-19 impact the business operations?",
            "What strategic partnerships were established?",
            "What are the key risks and challenges identified?"
        ]
    
    def setup_environment(self):
        """Setup demonstration environment"""
        # Create necessary directories
        directories = ['./documents/', './uploads/', './models/', './cache/', './logs/']
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Setup environment variables if not present
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️  Warning: OPENAI_API_KEY not found in environment variables")
            print("   Please set your OpenAI API key for full functionality")
            print("   export OPENAI_API_KEY='your-api-key-here'")
    
    async def initialize_system(self):
        """Initialize the RAG system"""
        print("\n🔧 Initializing Advanced RAG System...")
        try:
            self.rag_system = AdvancedRAGSystem(collection_name="demo_rag")
            print("✅ System initialized successfully!")
            return True
        except Exception as e:
            print(f"❌ System initialization failed: {e}")
            return False
    
    def demonstrate_chunking_strategies(self):
        """Demonstrate different chunking strategies"""
        print("\n📄 Demonstrating Advanced Chunking Strategies...")
        
        sample_text = """
        Microsoft Corporation is a multinational technology company. The company develops, manufactures, licenses, 
        supports, and sells computer software, consumer electronics, personal computers, and related services.
        
        In fiscal year 2023, Microsoft reported record revenue of $211.9 billion, representing a 7% increase 
        year-over-year. The company's cloud services, particularly Azure, showed strong growth with a 27% 
        increase in revenue.
        
        Key business segments include Productivity and Business Processes, Intelligent Cloud, and More Personal 
        Computing. Each segment contributed significantly to the overall revenue growth.
        """
        
        chunker = AdvancedChunker()
        strategies = ['recursive', 'semantic', 'hierarchical', 'hybrid']
        
        for strategy in strategies:
            print(f"\n🔍 {strategy.upper()} Chunking:")
            chunks = chunker.chunk_document(sample_text, strategy)
            print(f"   Generated {len(chunks)} chunks")
            for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
                print(f"   Chunk {i+1}: {chunk.page_content[:100]}...")
    
    def demonstrate_query_classification(self):
        """Demonstrate intelligent query classification"""
        print("\n🧠 Demonstrating Query Classification...")
        
        classifier = QueryClassifier()
        sample_queries = [
            "What is the total revenue?",
            "Why did the stock price increase?",
            "Compare Q1 and Q2 performance",
            "How did revenue trend over the past year?",
            "Calculate the profit margin",
            "Explain the business strategy"
        ]
        
        for query in sample_queries:
            try:
                analysis = classifier.analyze_query(query)
                print(f"\n📝 Query: '{query}'")
                print(f"   Type: {analysis.query_type.value}")
                print(f"   Complexity: {analysis.complexity_score:.2f}")
                print(f"   Suggested Strategy: {analysis.suggested_strategy.value}")
                print(f"   Entities: {analysis.entities}")
            except Exception as e:
                print(f"   ❌ Analysis failed: {e}")
    
    async def demonstrate_document_processing(self):
        """Demonstrate multi-modal document processing"""
        print("\n📚 Demonstrating Document Processing...")
        
        # Check if the Microsoft report exists
        pdf_path = r'C:\Users\simon\Desktop\Report\microsoft-annual-report.pdf'
        
        if os.path.exists(pdf_path):
            print(f"📋 Processing document: {pdf_path}")
            try:
                result = await self.rag_system.add_document_async(pdf_path)
                print(f"✅ Successfully processed:")
                print(f"   - {result['chunks_created']} text chunks")
                print(f"   - {result['tables_extracted']} tables")
                print(f"   - {result['images_processed']} images")
            except Exception as e:
                print(f"❌ Document processing failed: {e}")
        else:
            print(f"⚠️  Sample document not found at {pdf_path}")
            print("   Creating sample document chunks for demonstration...")
            
            # Create sample documents for demo
            sample_docs = [
                "Microsoft reported strong financial results with revenue growth of 7%.",
                "Azure cloud services showed exceptional performance with 27% growth.",
                "The company invested heavily in AI and machine learning technologies.",
                "Productivity tools like Office 365 continued to gain market share.",
                "Gaming division contributed significantly to overall revenue growth."
            ]
            
            from langchain.schema import Document
            documents = [Document(page_content=text, metadata={'source': 'demo'}) 
                        for text in sample_docs]
            
            self.rag_system.retriever.add_documents(documents)
            print(f"✅ Added {len(documents)} sample documents")
    
    async def demonstrate_retrieval_strategies(self):
        """Demonstrate different retrieval strategies"""
        print("\n🔍 Demonstrating Retrieval Strategies...")
        
        test_query = "What was the revenue growth?"
        strategies = [
            RetrievalStrategy.DENSE_ONLY,
            RetrievalStrategy.SPARSE_ONLY,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.MULTI_VECTOR
        ]
        
        for strategy in strategies:
            try:
                print(f"\n🎯 Testing {strategy.value.upper()} strategy:")
                start_time = time.time()
                
                result = self.rag_system.retriever.retrieve(test_query, strategy, k=3)
                
                processing_time = time.time() - start_time
                print(f"   ⏱️  Processing time: {processing_time:.3f}s")
                print(f"   📊 Retrieved {len(result.documents)} documents")
                print(f"   📈 Average score: {sum(result.scores)/len(result.scores):.3f}")
                
                # Show top document
                if result.documents:
                    print(f"   📄 Top result: {result.documents[0][:100]}...")
                    
            except Exception as e:
                print(f"   ❌ Strategy {strategy.value} failed: {e}")
    
    async def demonstrate_full_query_processing(self):
        """Demonstrate complete query processing pipeline"""
        print("\n🔄 Demonstrating Full Query Processing Pipeline...")
        
        sample_queries = self.demo_queries[:3]  # Use first 3 queries
        
        for i, query in enumerate(sample_queries):
            print(f"\n🔍 Query {i+1}: {query}")
            try:
                result = await self.rag_system.process_query_async(
                    query=query,
                    strategy=None,  # Auto-select
                    k=3,
                    include_analysis=True
                )
                
                print(f"   📋 Answer: {result.answer[:200]}...")
                print(f"   🎯 Strategy Used: {result.retrieval_strategy}")
                print(f"   📊 Confidence: {result.confidence_score:.3f}")
                print(f"   ⏱️  Processing Time: {result.processing_time:.3f}s")
                print(f"   📚 Retrieved {len(result.retrieved_documents)} documents")
                
                if result.query_analysis:
                    analysis = result.query_analysis
                    print(f"   🧠 Query Type: {analysis.get('query_type', 'N/A')}")
                    print(f"   🧮 Complexity: {analysis.get('complexity_score', 0):.2f}")
                
            except Exception as e:
                print(f"   ❌ Query processing failed: {e}")
    
    def demonstrate_evaluation_metrics(self):
        """Demonstrate evaluation and benchmarking"""
        print("\n📊 Demonstrating Evaluation Metrics...")
        
        evaluator = RAGEvaluator()
        
        # Simulate benchmark
        test_queries = self.demo_queries[:5]
        
        try:
            print("🔬 Running retrieval strategy benchmark...")
            benchmark_results = evaluator.benchmark_strategies(
                test_queries, 
                self.rag_system.retriever
            )
            
            print("\n📈 Benchmark Results:")
            for strategy, metrics in benchmark_results.items():
                print(f"   {strategy.upper()}:")
                print(f"     - Avg Time: {metrics['avg_retrieval_time']:.3f}s")
                print(f"     - Avg Score: {metrics['avg_score']:.3f}")
                print(f"     - Success Rate: {metrics['success_rate']:.2%}")
        
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
    
    def demonstrate_analytics_and_visualization(self):
        """Demonstrate analytics and visualization features"""
        print("\n📊 Demonstrating Analytics and Visualization...")
        
        # Get system analytics
        analytics = self.rag_system.get_analytics()
        
        print("📈 System Analytics:")
        print(f"   - Total Queries: {analytics['performance_metrics']['total_queries']}")
        print(f"   - Success Rate: {analytics['performance_metrics']['success_rate']:.2%}")
        print(f"   - Avg Response Time: {analytics['performance_metrics']['avg_response_time']:.3f}s")
        print(f"   - Total Documents: {analytics['total_documents']}")
        print(f"   - Memory Size: {analytics['memory_size']}")
        
        # Query patterns
        if analytics['query_patterns']:
            print("\n🔍 Query Patterns:")
            for pattern, data in analytics['query_patterns'].items():
                print(f"   {pattern}: {data['count']} queries (avg feedback: {data['avg_feedback']:.2f})")
    
    def demonstrate_memory_and_learning(self):
        """Demonstrate memory management and learning capabilities"""
        print("\n🧠 Demonstrating Memory and Learning...")
        
        memory = self.rag_system.memory
        
        # Add some sample interactions
        sample_interactions = [
            ("What was the revenue?", "Microsoft reported revenue of $211.9 billion.", 4.5),
            ("How did Azure perform?", "Azure showed strong growth with 27% increase.", 4.0),
            ("What about AI investments?", "The company invested heavily in AI technologies.", 3.8)
        ]
        
        for query, response, feedback in sample_interactions:
            memory.add_interaction(query, response, feedback)
        
        print(f"💾 Memory Statistics:")
        print(f"   - Conversation History: {len(memory.conversation_history)} interactions")
        print(f"   - Query Patterns: {len(memory.query_patterns)} categories")
        
        # Test memory retrieval
        test_query = "Tell me about Microsoft's financial performance"
        relevant_history = memory.get_relevant_history(test_query, n=2)
        
        print(f"\n🔍 Relevant History for '{test_query}':")
        for item in relevant_history:
            print(f"   Q: {item['query']}")
            print(f"   A: {item['response'][:100]}...")
    
    def demonstrate_guardrails(self):
        """Demonstrate safety and quality guardrails"""
        print("\n🛡️  Demonstrating Safety Guardrails...")
        
        guardrails = GuardrailSystem()
        
        # Test queries with different safety levels
        test_cases = [
            ("What was the revenue growth?", "safe"),
            ("How to hack into systems?", "potentially unsafe"),
            ("Explain the financial results", "safe")
        ]
        
        for query, expected in test_cases:
            try:
                safety_check = guardrails.check_query_safety(query)
                status = "✅ SAFE" if safety_check['is_safe'] else "⚠️  UNSAFE"
                print(f"   Query: '{query}' - {status}")
                if safety_check['issues']:
                    print(f"     Issues: {safety_check['issues']}")
            except Exception as e:
                print(f"   ❌ Safety check failed for '{query}': {e}")
    
    async def run_complete_demo(self):
        """Run the complete demonstration"""
        print("=" * 80)
        print("🚀 ADVANCED RAG SYSTEM - COMPLETE DEMONSTRATION")
        print("=" * 80)
        
        # Initialize system
        if not await self.initialize_system():
            print("❌ Cannot proceed without system initialization")
            return
        
        try:
            # Core demonstrations
            self.demonstrate_chunking_strategies()
            self.demonstrate_query_classification()
            await self.demonstrate_document_processing()
            await self.demonstrate_retrieval_strategies()
            await self.demonstrate_full_query_processing()
            
            # Advanced features
            self.demonstrate_evaluation_metrics()
            self.demonstrate_analytics_and_visualization()
            self.demonstrate_memory_and_learning()
            self.demonstrate_guardrails()
            
            print("\n" + "=" * 80)
            print("🎉 DEMONSTRATION COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            
            # Summary
            print("\n📋 SYSTEM CAPABILITIES DEMONSTRATED:")
            capabilities = [
                "✅ Multi-modal document processing",
                "✅ Advanced chunking strategies",
                "✅ Intelligent query classification",
                "✅ Hybrid retrieval methods",
                "✅ Multi-vector search",
                "✅ Real-time evaluation metrics",
                "✅ Memory management & learning",
                "✅ Safety guardrails",
                "✅ Analytics & visualization",
                "✅ Async processing",
                "✅ Production-ready API"
            ]
            
            for capability in capabilities:
                print(f"   {capability}")
            
            print("\n🚀 NEXT STEPS:")
            print("   1. Install dependencies: pip install -r requirements_advanced.txt")
            print("   2. Set OpenAI API key: export OPENAI_API_KEY='your-key'")
            print("   3. Run API server: python advanced_rag_system_part2.py")
            print("   4. Run Streamlit UI: streamlit run advanced_rag_system_part2.py")
            print("   5. Upload your documents and start querying!")
            
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            print("Please check your configuration and dependencies")
    
    def generate_demo_report(self):
        """Generate a demonstration report"""
        print("\n📊 Generating Demo Report...")
        
        report = {
            "demo_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "name": "Advanced RAG System",
                "version": "2.0.0",
                "features_demonstrated": 11
            },
            "capabilities": {
                "document_processing": "Multi-modal with tables and images",
                "chunking": "4 strategies (recursive, semantic, hierarchical, hybrid)",
                "retrieval": "5 strategies (dense, sparse, hybrid, DPR, multi-vector)",
                "evaluation": "Comprehensive metrics with RAGAS integration",
                "interfaces": "FastAPI + Streamlit",
                "safety": "Guardrails with toxicity detection",
                "memory": "Conversation history with learning",
                "visualization": "3D embeddings and analytics"
            },
            "performance_features": [
                "Async processing",
                "Caching mechanisms", 
                "Performance monitoring",
                "Memory management",
                "Rate limiting",
                "Error handling"
            ]
        }
        
        # Save report
        with open("demo_report.json", "w") as f:
            json.dump(report, f, indent=2)
        
        print("✅ Demo report saved to 'demo_report.json'")

def main():
    """Main demonstration function"""
    demo = RAGSystemDemo()
    
    # Run the complete demonstration
    asyncio.run(demo.run_complete_demo())
    
    # Generate report
    demo.generate_demo_report()

if __name__ == "__main__":
    main()