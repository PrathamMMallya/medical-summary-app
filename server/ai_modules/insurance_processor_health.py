# ai_modules/insurance_processor.py
import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path

# Add the Django project to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Django setup
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'core.settings')
import django
django.setup()

from insurance.models import InsuranceDocument, DocumentChunk, InsuranceQuery

# Your existing imports
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
load_dotenv()

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from docling.document_converter import DocumentConverter
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InsuranceRAGProcessor:
    """Enhanced RAG processor with Django integration"""
    
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model='nomic-embed-text:v1.5', 
            base_url="http://localhost:11434"
        )
        self.vector_stores = {}
        self.retrievers = {}
        self.rag_chain = None
        self.keyword_chain = None
        
    def load_and_convert_document(self, file_path: str) -> str:
        """Load and convert document to markdown"""
        try:
            logger.info(f"Converting document: {file_path}")
            converter = DocumentConverter()
            result = converter.convert(file_path)
            return result.document.export_to_markdown()
        except Exception as e:
            logger.error(f"Error converting document: {e}")
            raise
    
    def preprocess_markdown_content(self, markdown_content: str) -> str:
        """Clean and preprocess markdown content for better chunking"""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', markdown_content)
        
        # Ensure policy entries are properly separated
        content = re.sub(r'(\d+\.\s+[A-Za-z]+.*?)\n([A-Z])', r'\1\n\n\2', content)
        
        # Add markers for better chunking
        content = re.sub(r'(\d+\.\s+[A-Za-z][^0-9]*(?:Age|Diseases|Reimbursement|Premium|Highlights).*?)(?=\n\d+\.|\n##|\Z)', 
                         r'--- POLICY START ---\n\1\n--- POLICY END ---', content, flags=re.DOTALL)
        
        return content
    
    def get_optimized_chunk_strategies(self, markdown_content: str) -> Tuple[List[Document], List[Document], List[Document]]:
        """Create optimized chunking strategies that preserve policy information"""
        
        # Preprocess content
        processed_content = self.preprocess_markdown_content(markdown_content)
        
        # Strategy 1: Policy-based chunking
        policy_chunks = []
        policy_pattern = r'--- POLICY START ---\n(.*?)\n--- POLICY END ---'
        policies = re.findall(policy_pattern, processed_content, re.DOTALL)
        
        for i, policy in enumerate(policies):
            if len(policy.strip()) > 50:
                doc = Document(
                    page_content=policy.strip(),
                    metadata={'strategy': 'policy', 'policy_id': i}
                )
                policy_chunks.append(doc)
        
        # Strategy 2: Semantic chunks
        semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
            separators=["\n--- POLICY END ---", "\n\n", "\n", ". ", " "]
        )
        semantic_chunks = semantic_splitter.create_documents([processed_content])
        
        # Strategy 3: Header-based chunks
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
        header_chunks = markdown_splitter.split_text(processed_content)
        
        # Add strategy labels
        for chunk in semantic_chunks:
            chunk.metadata['strategy'] = 'semantic'
        for chunk in header_chunks:
            chunk.metadata['strategy'] = 'header'
        
        return policy_chunks, semantic_chunks, header_chunks
    
    def save_chunks_to_database(self, document_id: int, policy_chunks: List[Document], 
                               semantic_chunks: List[Document], header_chunks: List[Document]) -> None:
        """Save chunks to Django database"""
        try:
            document = InsuranceDocument.objects.get(id=document_id)
            
            # Clear existing chunks
            DocumentChunk.objects.filter(document=document).delete()
            
            all_chunks = [
                ('policy', policy_chunks),
                ('semantic', semantic_chunks),
                ('header', header_chunks)
            ]
            
            chunk_counter = 0
            for strategy, chunks in all_chunks:
                for chunk in chunks:
                    # Generate embedding
                    embedding = self.embeddings.embed_query(chunk.page_content)
                    
                    # Create chunk record
                    DocumentChunk.objects.create(
                        document=document,
                        chunk_id=f"{strategy}_{chunk_counter}",
                        content=chunk.page_content,
                        strategy=strategy,
                        metadata=chunk.metadata,
                        embedding_vector=embedding
                    )
                    chunk_counter += 1
            
            # Update document
            document.total_chunks = chunk_counter
            document.processed = True
            document.save()
            
            logger.info(f"Saved {chunk_counter} chunks to database for document {document_id}")
            
        except Exception as e:
            logger.error(f"Error saving chunks to database: {e}")
            raise
    
    def load_chunks_from_database(self, document_id: int = None) -> List[Document]:
        """Load chunks from Django database"""
        try:
            if document_id:
                chunks = DocumentChunk.objects.filter(document_id=document_id)
            else:
                chunks = DocumentChunk.objects.all()
            
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk.content,
                    metadata={
                        'strategy': chunk.strategy,
                        'chunk_id': chunk.chunk_id,
                        'document_id': chunk.document.id,
                        **chunk.metadata
                    }
                )
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} chunks from database")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading chunks from database: {e}")
            raise
    
    def setup_vector_stores_from_db(self, document_id: int = None) -> Dict:
        """Create vector stores from database chunks"""
        try:
            # Load chunks from database
            all_chunks = self.load_chunks_from_database(document_id)
            
            if not all_chunks:
                logger.warning("No chunks found in database")
                return {}
            
            # Create embeddings for vector store
            embeddings_list = []
            for chunk in all_chunks:
                # Try to get embedding from database first
                db_chunk = DocumentChunk.objects.get(
                    document_id=chunk.metadata.get('document_id'),
                    chunk_id=chunk.metadata.get('chunk_id')
                )
                
                if db_chunk.embedding_vector:
                    embeddings_list.append(db_chunk.embedding_vector)
                else:
                    # Generate embedding if not stored
                    embedding = self.embeddings.embed_query(chunk.page_content)
                    embeddings_list.append(embedding)
                    # Save it back to database
                    db_chunk.embedding_vector = embedding
                    db_chunk.save()
            
            # Create FAISS index
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            index = faiss.IndexFlatL2(embeddings_array.shape[1])
            index.add(embeddings_array)
            
            # Create vector store
            vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore({i: doc for i, doc in enumerate(all_chunks)}),
                index_to_docstore_id={i: i for i in range(len(all_chunks))}
            )
            
            stores = {'main': vector_store}
            
            # Create policy-specific store
            policy_chunks = [doc for doc in all_chunks if doc.metadata.get('strategy') == 'policy']
            if policy_chunks:
                policy_embeddings = [embeddings_list[i] for i, doc in enumerate(all_chunks) 
                                   if doc.metadata.get('strategy') == 'policy']
                policy_array = np.array(policy_embeddings, dtype=np.float32)
                policy_index = faiss.IndexFlatL2(policy_array.shape[1])
                policy_index.add(policy_array)
                
                policy_store = FAISS(
                    embedding_function=self.embeddings,
                    index=policy_index,
                    docstore=InMemoryDocstore({i: doc for i, doc in enumerate(policy_chunks)}),
                    index_to_docstore_id={i: i for i in range(len(policy_chunks))}
                )
                stores['policy'] = policy_store
            
            self.vector_stores = stores
            logger.info(f"Created vector stores with {len(all_chunks)} total chunks")
            return stores
            
        except Exception as e:
            logger.error(f"Error setting up vector stores: {e}")
            raise
    
    def create_smart_retrievers(self) -> Dict:
        """Create smart retrievers with different search strategies"""
        if not self.vector_stores:
            raise ValueError("Vector stores not initialized. Call setup_vector_stores_from_db first.")
        
        retrievers = {}
        
        # Primary retriever
        retrievers['primary'] = self.vector_stores['main'].as_retriever(
            search_type="similarity",
            search_kwargs={'k': 6}
        )
        
        # Diverse retriever
        retrievers['diverse'] = self.vector_stores['main'].as_retriever(
            search_type="mmr",
            search_kwargs={'k': 4, 'fetch_k': 10, 'lambda_mult': 0.3}
        )
        
        # Policy-specific retriever
        if 'policy' in self.vector_stores:
            retrievers['policy'] = self.vector_stores['policy'].as_retriever(
                search_type="similarity",
                search_kwargs={'k': 5}
            )
        
        self.retrievers = retrievers
        return retrievers
    
    def intelligent_hybrid_retrieve(self, question: str, top_k: int = 5) -> List[Document]:
        """Intelligent hybrid retrieval with deduplication and ranking"""
        if not self.retrievers:
            raise ValueError("Retrievers not initialized. Call create_smart_retrievers first.")

        all_docs = []
        seen_content = set()
        
        question_lower = question.lower()
        key_terms = set()  # Use set to avoid duplicates

        # === Manual extraction from question ===
        age_match = re.search(r'age[:\s]*(\d+)', question_lower)
        if age_match:
            key_terms.add(f"age {age_match.group(1)}")
        
        conditions = ['diabetes', 'heart', 'cancer', 'hypertension', 'kidney', 'liver']
        for condition in conditions:
            if condition in question_lower:
                key_terms.add(condition)
        
        budget_match = re.search(r'₹(\d+,?\d*)', question)
        if budget_match:
            key_terms.add(f"budget {budget_match.group(1)}")

        # === Model-based structured keyword extraction ===
        extractor = self.create_keyword_extractor_chain()
        structured_keywords = extractor.invoke({"question": question_lower})
        print(structured_keywords)  # You can log this instead if needed

        for item in structured_keywords:
            item_lower = item.lower()

            if "age" in item_lower:
                match = re.search(r'age[:\s]*(\d+)', item_lower)
                if match:
                    key_terms.add(f"age {match.group(1)}")

            elif "health condition" in item_lower:
                parts = item_lower.split("health conditions:")[-1].split(",")
                for cond in parts:
                    key_terms.add(cond.strip())

            elif "budget" in item_lower:
                match = re.search(r'₹(\d+,?\d*)', item_lower)
                if match:
                    key_terms.add(f"budget {match.group(1)}")

            elif "coverage" in item_lower:
                parts = item_lower.split(":")[-1].split(",")
                for coverage in parts:
                    key_terms.add(coverage.strip())

        key_terms = list(key_terms)  # Convert to list for scoring
        logger.info(f"Extracted key terms: {key_terms}")

        # === Retrieval and ranking ===
        for name, retriever in self.retrievers.items():
            try:
                docs = retriever.invoke(question)
                for doc in docs:
                    content_signature = ' '.join(doc.page_content.split()[:20])
                    content_hash = hash(content_signature)

                    if content_hash not in seen_content:
                        relevance_score = 0
                        doc_content_lower = doc.page_content.lower()

                        for term in key_terms:
                            if term in doc_content_lower:
                                relevance_score += 1

                        doc.metadata['retriever'] = name
                        doc.metadata['relevance_score'] = relevance_score
                        all_docs.append(doc)
                        seen_content.add(content_hash)

            except Exception as e:
                logger.error(f"Error with {name} retriever: {e}")

        all_docs.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        return all_docs[:top_k]

    
    def create_rag_chain(self):
        """Create RAG chain with enhanced prompting"""
        if not self.retrievers:
            raise ValueError("Retrievers not initialized.")
        
        prompt = """
            You are an expert health insurance advisor in India. Analyze the provided insurance policies and recommend the most suitable options based on the user's specific needs.

            USER REQUIREMENTS:
            {question}

            AVAILABLE INSURANCE POLICIES:
            {context}

            INSTRUCTIONS:
            1. Carefully analyze each policy option provided
            2. Match the user's age, health condition, and budget with suitable policies
            3. Consider premium affordability (monthly income vs annual premium)
            4. Ensure the health condition is covered under the policy
            5. Provide 2-3 specific policy recommendations with clear reasoning
            6. If no suitable policy exists, explain why and suggest alternatives

            RESPONSE FORMAT:
            ## Recommended Insurance Policies

            ### Policy 1: [Policy Name]
            - **Why suitable**: [Specific reasons]
            - **Coverage**: [What's covered for user's condition]
            - **Premium**: [Annual premium and monthly breakdown]
            - **Key Benefits**: [Relevant benefits]

            ### Policy 2: [Policy Name]
            - **Why suitable**: [Specific reasons]
            - **Coverage**: [What's covered for user's condition]
            - **Premium**: [Annual premium and monthly breakdown]
            - **Key Benefits**: [Relevant benefits]

            ## Summary
            [Brief summary of why these policies are recommended]

            Base your recommendations ONLY on the provided policy information.
            """
        
        # Try to load the best available model
        model_options = ["phi3:mini", "llama3", "llama2"]


        
        model = None
        for model_name in model_options:
            try:
                model = ChatOllama(
                    model=model_name, 
                    base_url="http://localhost:11434",
                    temperature=0.1,
                    num_predict=1000
                )
                logger.info(f"Successfully loaded model: {model_name}")
                break
            except Exception as e:
                logger.error(f"Failed to load {model_name}: {e}")
                continue
        
        if not model:
            raise Exception("No suitable model found. Please ensure Ollama is running.")
        
        prompt_template = ChatPromptTemplate.from_template(prompt)
        
        def retrieval_chain(question):
            docs = self.intelligent_hybrid_retrieve(question, top_k=5)
            context = self.format_context(docs)
            return {"context": context, "question": question}
        
        self.rag_chain = (
            retrieval_chain
            | prompt_template
            | model
            | StrOutputParser()
        )
        
        return self.rag_chain
    def create_keyword_extractor_chain(self):
        """Creates a LLaMA-powered chain to extract structured keywords from user input"""

        model = ChatOllama(
            model="llama3.2:3b",
            base_url="http://localhost:11434",
            temperature=0.1
        )

        prompt = """
    You are a helpful assistant. Extract the following from the user's message:
    1. Age
    2. Health conditions (like asthma, thyroid, etc.)
    3. Budget or financial constraints the exact value (like 15000rs ,etc. if the value is given) 
    4. Desired coverage (doctor visits, prescriptions, etc.)

    Input:
    {question}

    Return the extracted information as a Python list of strings.
    """

        prompt_template = ChatPromptTemplate.from_template(prompt)

        self.keyword_chain = (
            prompt_template
            | model
            | StrOutputParser()
        )

        return self.keyword_chain

    
    def format_context(self, docs: List[Document]) -> str:
        """Format documents for context"""
        if not docs:
            return "No relevant insurance policies found."
        
        formatted_parts = []
        for i, doc in enumerate(docs, 1):
            retriever_info = doc.metadata.get('retriever', 'unknown')
            strategy_info = doc.metadata.get('strategy', 'unknown')
            relevance_score = doc.metadata.get('relevance_score', 0)
            
            # Clean content
            content = doc.page_content.strip()
            content = re.sub(r'--- POLICY (START|END) ---', '', content).strip()
            
            formatted_parts.append(
                f"=== POLICY OPTION {i} (via {retriever_info}-{strategy_info}, relevance: {relevance_score}) ===\n"
                f"{content}\n"
            )
        
        return "\n".join(formatted_parts)
    
    def process_document(self, file_path: str, document_id: int) -> bool:
        """Process a document and save chunks to database"""
        try:
            # Convert document
            markdown_content = self.load_and_convert_document(file_path)
            
            # Create chunks
            policy_chunks, semantic_chunks, header_chunks = self.get_optimized_chunk_strategies(markdown_content)
            
            # Save to database
            self.save_chunks_to_database(document_id, policy_chunks, semantic_chunks, header_chunks)
            
            logger.info(f"Successfully processed document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            return False
    
    def initialize_system(self, document_id: int = None) -> bool:
        """Initialize the RAG system from database"""
        try:
            # Setup vector stores
            self.setup_vector_stores_from_db(document_id)
            
            # Create retrievers
            self.create_smart_retrievers()
            
            # Create RAG chain
            self.create_rag_chain()
            
            logger.info("RAG system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            return False
    
    def query_insurance(self, question: str, save_to_db: bool = True) -> str:
        """Query the insurance system and optionally save to database"""
        if not self.rag_chain:
            raise ValueError("RAG system not initialized. Call initialize_system first.")
        
        try:
            start_time = time.time()
            
            # Get retrieved documents for tracking
            retrieved_docs = self.intelligent_hybrid_retrieve(question, top_k=5)
            chunk_ids = [doc.metadata.get('chunk_id', '') for doc in retrieved_docs]
            
            # Generate response
            response = self.rag_chain.invoke(question)
            
            processing_time = time.time() - start_time
            
            # Save query to database
            if save_to_db:
                InsuranceQuery.objects.create(
                    query_text=question,
                    response_text=response,
                    retrieved_chunks=chunk_ids,
                    processing_time=processing_time
                )
            
            logger.info(f"Query processed in {processing_time:.2f} seconds")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
    
    @staticmethod
    def clear_all_data():
        """Clear all data from database"""
        try:
            DocumentChunk.objects.all().delete()
            InsuranceDocument.objects.all().delete()
            InsuranceQuery.objects.all().delete()
            logger.info("All data cleared from database")
            return True
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            return False