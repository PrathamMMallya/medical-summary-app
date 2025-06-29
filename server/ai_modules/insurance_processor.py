# ai_modules/insurance_processor.py
# ai_modules/insurance_processor.py
import os
import sys
import json
import time
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
from pathlib import Path
import faiss  # Add this import
from insurance.models import InsuranceDocument, DocumentChunk, InsuranceQuery, INSURANCE_TYPES
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from docling.document_converter import DocumentConverter
import re
logger = logging.getLogger(__name__)

class InsuranceRAGProcessor:
    """Enhanced RAG processor with Django integration and multi-insurance type support"""
    
    PROMPT_TEMPLATES = {
        'health': """
            You are an expert health insurance advisor in India. Analyze the provided health insurance policies and recommend the most suitable options based on the user's specific needs.

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
            ## Recommended Health Insurance Policies

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
        """,
        'car': """
            You are an expert car insurance advisor in India. Analyze the provided car insurance policies and recommend the most suitable options based on the user's specific needs.

            USER REQUIREMENTS:
            {question}

            AVAILABLE INSURANCE POLICIES:
            {context}

            INSTRUCTIONS:
            1. Carefully analyze each policy option provided
            2. Match the user's vehicle details (make, model, year, fuel type), driving history, and budget with suitable policies
            3. Consider premium affordability and coverage type (comprehensive or third-party)
            4. Ensure the policy covers the user's requirements (e.g., accident coverage, theft, natural disasters)
            5. Provide 2-3 specific policy recommendations with clear reasoning
            6. If no suitable policy exists, explain why and suggest alternatives

            RESPONSE FORMAT:
            ## Recommended Car Insurance Policies

            ### Policy 1: [Policy Name]
            - **Why suitable**: [Specific reasons]
            - **Coverage**: [What's covered for the vehicle]
            - **Premium**: [Annual premium and monthly breakdown]
            - **Key Benefits**: [Relevant benefits, e.g., no-claim bonus, roadside assistance]

            ### Policy 2: [Policy Name]
            - **Why suitable**: [Specific reasons]
            - **Coverage**: [What's covered for the vehicle]
            - **Premium**: [Annual premium and monthly breakdown]
            - **Key Benefits**: [Relevant benefits]

            ## Summary
            [Brief summary of why these policies are recommended]
        """,
        'life': """
            You are an expert life insurance advisor in India. Analyze the provided life insurance policies and recommend the most suitable options based on the user's specific needs.

            USER REQUIREMENTS:
            {question}

            AVAILABLE INSURANCE POLICIES:
            {context}

            INSTRUCTIONS:
            1. Carefully analyze each policy option provided
            2. Match the user's age, income, family status, and coverage needs with suitable policies
            3. Consider premium affordability and policy term
            4. Ensure the policy covers the user's requirements (e.g., term insurance, endowment plans)
            5. Provide 2-3 specific policy recommendations with clear reasoning
            6. If no suitable policy exists, explain why and suggest alternatives

            RESPONSE FORMAT:
            ## Recommended Life Insurance Policies

            ### Policy 1: [Policy Name]
            - **Why suitable**: [Specific reasons]
            - **Coverage**: [What's covered, e.g., sum assured, riders]
            - **Premium**: [Annual premium and monthly breakdown]
            - **Key Benefits**: [Relevant benefits]

            ### Policy 2: [Policy Name]
            - **Why suitable**: [Specific reasons]
            - **Coverage**: [What's covered]
            - **Premium**: [Annual premium and monthly breakdown]
            - **Key Benefits**: [Relevant benefits]

            ## Summary
            [Brief summary of why these policies are recommended]
        """,
        'home': """
            You are an expert home insurance advisor in India. Analyze the provided home insurance policies and recommend the most suitable options based on the user's specific needs.

            USER REQUIREMENTS:
            {question}

            AVAILABLE INSURANCE POLICIES:
            {context}

            INSTRUCTIONS:
            1. Carefully analyze each policy option provided
            2. Match the user's property details (location, type, value) and coverage needs with suitable policies
            3. Consider premium affordability and coverage scope (e.g., structure, contents, natural disasters)
            4. Ensure the policy covers the user's requirements
            5. Provide 2-3 specific policy recommendations with clear reasoning
            6. If no suitable policy exists, explain why and suggest alternatives

            RESPONSE FORMAT:
            ## Recommended Home Insurance Policies

            ### Policy 1: [Policy Name]
            - **Why suitable**: [Specific reasons]
            - **Coverage**: [What's covered, e.g., structure, contents]
            - **Premium**: [Annual premium and monthly breakdown]
            - **Key Benefits**: [Relevant benefits]

            ### Policy 2: [Policy Name]
            - **Why suitable**: [Specific reasons]
            - **Coverage**: [What's covered]
            - **Premium**: [Annual premium and monthly breakdown]
            - **Key Benefits**: [Relevant benefits]

            ## Summary
            [Brief summary of why these policies are recommended]
        """
    }

    def __init__(self, insurance_type='health'):
        if insurance_type not in dict(INSURANCE_TYPES):
            raise ValueError(f"Invalid insurance type: {insurance_type}")
        self.insurance_type = insurance_type
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
            logger.info(f"Converting {self.insurance_type} document: {file_path}")
            converter = DocumentConverter()
            result = converter.convert(file_path)
            return result.document.export_to_markdown()
        except Exception as e:
            logger.error(f"Error converting {self.insurance_type} document: {e}")
            raise

    def preprocess_markdown_content(self, markdown_content: str) -> str:
        """Clean and preprocess markdown content for better chunking"""
        content = re.sub(r'\n\s*\n\s*\n', '\n\n', markdown_content)
        
        if self.insurance_type == 'car':
            content = re.sub(r'(\d+\.\s+[A-Za-z]+.*?)(?=\n\d+\.|\n##|\Z)', 
                            r'--- CAR POLICY START ---\n\1\n--- CAR POLICY END ---', content, flags=re.DOTALL)
        elif self.insurance_type == 'health':
            content = re.sub(r'(\d+\.\s+[A-Za-z]+.*?)(?=\n\d+\.|\n##|\Z)', 
                            r'--- HEALTH POLICY START ---\n\1\n--- HEALTH POLICY END ---', content, flags=re.DOTALL)
        elif self.insurance_type == 'life':
            content = re.sub(r'(\d+\.\s+[A-Za-z]+.*?)(?=\n\d+\.|\n##|\Z)', 
                            r'--- LIFE POLICY START ---\n\1\n--- LIFE POLICY END ---', content, flags=re.DOTALL)
        elif self.insurance_type == 'home':
            content = re.sub(r'(\d+\.\s+[A-Za-z]+.*?)(?=\n\d+\.|\n##|\Z)', 
                            r'--- HOME POLICY START ---\n\1\n--- HOME POLICY END ---', content, flags=re.DOTALL)
        
        return content

    def get_optimized_chunk_strategies(self, markdown_content: str) -> Tuple[List[Document], List[Document], List[Document]]:
        """Create optimized chunking strategies that preserve policy information"""
        processed_content = self.preprocess_markdown_content(markdown_content)
        
        policy_chunks = []
        policy_pattern = rf'--- {self.insurance_type.upper()} POLICY START ---\n(.*?)\n--- {self.insurance_type.upper()} POLICY END ---'
        policies = re.findall(policy_pattern, processed_content, re.DOTALL)
        
        for i, policy in enumerate(policies):
            if len(policy.strip()) > 50:
                doc = Document(
                    page_content=policy.strip(),
                    metadata={'strategy': 'policy', 'policy_id': i, 'insurance_type': self.insurance_type}
                )
                policy_chunks.append(doc)
        
        semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=200,
            length_function=len,
            separators=[f"\n--- {self.insurance_type.upper()} POLICY END ---", "\n\n", "\n", ". ", " "]
        )
        semantic_chunks = semantic_splitter.create_documents([processed_content])
        
        headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
        header_chunks = markdown_splitter.split_text(processed_content)
        
        for chunk in semantic_chunks:
            chunk.metadata['strategy'] = 'semantic'
            chunk.metadata['insurance_type'] = self.insurance_type
        for chunk in header_chunks:
            chunk.metadata['strategy'] = 'header'
            chunk.metadata['insurance_type'] = self.insurance_type
        
        return policy_chunks, semantic_chunks, header_chunks

    def save_chunks_to_database(self, document_id: int, policy_chunks: List[Document], 
                               semantic_chunks: List[Document], header_chunks: List[Document]) -> None:
        """Save chunks to Django database"""
        try:
            document = InsuranceDocument.objects.get(id=document_id, insurance_type=self.insurance_type)
            DocumentChunk.objects.filter(document=document, insurance_type=self.insurance_type).delete()
            
            all_chunks = [
                ('policy', policy_chunks),
                ('semantic', semantic_chunks),
                ('header', header_chunks)
            ]
            
            chunk_counter = 0
            for strategy, chunks in all_chunks:
                for chunk in chunks:
                    embedding = self.embeddings.embed_query(chunk.page_content)
                    DocumentChunk.objects.create(
                        document=document,
                        chunk_id=f"{self.insurance_type}_{strategy}_{chunk_counter}",
                        content=chunk.page_content,
                        strategy=strategy,
                        metadata=chunk.metadata,
                        embedding_vector=embedding,
                        insurance_type=self.insurance_type
                    )
                    chunk_counter += 1
            
            document.total_chunks = chunk_counter
            document.processed = True
            document.save()
            
            logger.info(f"Saved {chunk_counter} {self.insurance_type} chunks to database for document {document_id}")
        except Exception as e:
            logger.error(f"Error saving {self.insurance_type} chunks to database: {e}")
            raise

    def load_chunks_from_database(self, document_id: int = None) -> List[Document]:
        """Load chunks from Django database"""
        try:
            if document_id:
                chunks = DocumentChunk.objects.filter(document_id=document_id, insurance_type=self.insurance_type)
            else:
                chunks = DocumentChunk.objects.filter(insurance_type=self.insurance_type)
            
            documents = []
            for chunk in chunks:
                doc = Document(
                    page_content=chunk.content,
                    metadata={
                        'strategy': chunk.strategy,
                        'chunk_id': chunk.chunk_id,
                        'document_id': chunk.document.id,
                        'insurance_type': chunk.insurance_type,
                        **chunk.metadata
                    }
                )
                documents.append(doc)
            
            logger.info(f"Loaded {len(documents)} {self.insurance_type} chunks from database")
            return documents
        except Exception as e:
            logger.error(f"Error loading {self.insurance_type} chunks from database: {e}")
            raise

    def setup_vector_stores_from_db(self, document_id: int = None) -> Dict:
        """Create vector stores from database chunks"""
        try:
            all_chunks = self.load_chunks_from_database(document_id)
            if not all_chunks:
                logger.warning(f"No {self.insurance_type} chunks found in database")
                return {}
            
            embeddings_list = []
            for chunk in all_chunks:
                db_chunk = DocumentChunk.objects.get(
                    document_id=chunk.metadata.get('document_id'),
                    chunk_id=chunk.metadata.get('chunk_id'),
                    insurance_type=self.insurance_type
                )
                if db_chunk.embedding_vector:
                    embeddings_list.append(db_chunk.embedding_vector)
                else:
                    embedding = self.embeddings.embed_query(chunk.page_content)
                    embeddings_list.append(embedding)
                    db_chunk.embedding_vector = embedding
                    db_chunk.save()
            
            embeddings_array = np.array(embeddings_list, dtype=np.float32)
            index = faiss.IndexFlatL2(embeddings_array.shape[1])
            index.add(embeddings_array)
            
            vector_store = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore({i: doc for i, doc in enumerate(all_chunks)}),
                index_to_docstore_id={i: i for i in range(len(all_chunks))}
            )
            
            stores = {'main': vector_store}
            
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
            logger.info(f"Created {self.insurance_type} vector stores with {len(all_chunks)} total chunks")
            return stores
        except Exception as e:
            logger.error(f"Error setting up {self.insurance_type} vector stores: {e}")
            raise

    def create_smart_retrievers(self) -> Dict:
        """Create smart retrievers with different search strategies"""
        if not self.vector_stores:
            raise ValueError(f"{self.insurance_type} vector stores not initialized.")
        
        retrievers = {}
        retrievers['primary'] = self.vector_stores['main'].as_retriever(
            search_type="similarity",
            search_kwargs={'k': 6}
        )
        retrievers['diverse'] = self.vector_stores['main'].as_retriever(
            search_type="mmr",
            search_kwargs={'k': 4, 'fetch_k': 10, 'lambda_mult': 0.3}
        )
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
            raise ValueError(f"{self.insurance_type} retrievers not initialized.")

        all_docs = []
        seen_content = set()
        
        question_lower = question.lower()
        key_terms = set()

        if self.insurance_type == 'car':
            vehicle_match = re.search(r'(suv|sedan|hatchback|bike|motorcycle|car|[a-zA-Z]+\s+[a-zA-Z0-9]+)', question_lower)
            if vehicle_match:
                key_terms.add(vehicle_match.group(0))
            
            year_match = re.search(r'(\d{4})', question_lower)
            if year_match:
                key_terms.add(f"year {year_match.group(0)}")
            
            coverage_types = ['comprehensive', 'third-party', 'third party', 'own-damage', 'theft', 'accident']
            for coverage in coverage_types:
                if coverage in question_lower:
                    key_terms.add(coverage)
        
        elif self.insurance_type == 'health':
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

        extractor = self.create_keyword_extractor_chain()
        structured_keywords = extractor.invoke({"question": question_lower})

        for item in structured_keywords:
            item_lower = item.lower()
            if self.insurance_type == 'car':
                if "vehicle" in item_lower:
                    key_terms.add(item_lower.split("vehicle:")[-1].strip())
                elif "coverage" in item_lower:
                    key_terms.add(item_lower.split("coverage:")[-1].strip())
            elif self.insurance_type == 'health':
                if "age" in item_lower:
                    match = re.search(r'age[:\s]*(\d+)', item_lower)
                    if match:
                        key_terms.add(f"age {match.group(1)}")
                elif "health condition" in item_lower:
                    parts = item_lower.split("health conditions:")[-1].split(",")
                    for cond in parts:
                        key_terms.add(cond.strip())
            elif self.insurance_type == 'life':
                if "sum assured" in item_lower:
                    key_terms.add(item_lower.split("sum assured:")[-1].strip())
                elif "term" in item_lower:
                    key_terms.add(item_lower.split("term:")[-1].strip())
            elif self.insurance_type == 'home':
                if "property" in item_lower:
                    key_terms.add(item_lower.split("property:")[-1].strip())
                elif "coverage" in item_lower:
                    key_terms.add(item_lower.split("coverage:")[-1].strip())

        key_terms = list(key_terms)
        logger.info(f"Extracted key terms for {self.insurance_type}: {key_terms}")

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
                logger.error(f"Error with {self.insurance_type} {name} retriever: {e}")

        all_docs.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
        return all_docs[:top_k]

    def create_rag_chain(self):
        """Create RAG chain with enhanced prompting"""
        if not self.retrievers:
            raise ValueError(f"{self.insurance_type} retrievers not initialized.")
        
        prompt_template = ChatPromptTemplate.from_template(self.PROMPT_TEMPLATES.get(self.insurance_type))
        
        model_options = ["llama3.2:3b", "llama3", "llama2"]
        model = None
        for model_name in model_options:
            try:
                model = ChatOllama(
                    model=model_name, 
                    base_url="http://localhost:11434",
                    temperature=0.1,
                    num_predict=1000
                )
                logger.info(f"Successfully loaded model for {self.insurance_type}: {model_name}")
                break
            except Exception as e:
                logger.error(f"Failed to load {model_name} for {self.insurance_type}: {e}")
                continue
        
        if not model:
            raise Exception(f"No suitable model found for {self.insurance_type}. Please ensure Ollama is running.")
        
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

        prompt_templates = {
            'car': """
                You are a helpful assistant. Extract the following from the user's car insurance query:
                1. Vehicle details (make, model, year, fuel type)
                2. Coverage preferences (comprehensive, third-party, etc.)
                3. Budget or financial constraints (exact value if given)
                4. Driving history or no-claim bonus status

                Input:
                {question}

                Return the extracted information as a Python list of strings.
            """,
            'health': """
                You are a helpful assistant. Extract the following from the user's health insurance query:
                1. Age
                2. Health conditions (like asthma, thyroid, etc.)
                3. Budget or financial constraints (exact value if given)
                4. Desired coverage (doctor visits, prescriptions, etc.)

                Input:
                {question}

                Return the extracted information as a Python list of strings.
            """,
            'life': """
                You are a helpful assistant. Extract the following from the user's life insurance query:
                1. Age
                2. Family status (dependents, marital status)
                3. Budget or financial constraints (exact value if given)
                4. Desired coverage (sum assured, term duration)

                Input:
                {question}

                Return the extracted information as a Python list of strings.
            """,
            'home': """
                You are a helpful assistant. Extract the following from the user's home insurance query:
                1. Property details (location, type, value)
                2. Coverage preferences (structure, contents, natural disasters)
                3. Budget or financial constraints (exact value if given)

                Input:
                {question}

                Return the extracted information as a Python list of strings.
            """
        }

        prompt_template = ChatPromptTemplate.from_template(prompt_templates.get(self.insurance_type))
        self.keyword_chain = (
            prompt_template
            | model
            | StrOutputParser()
        )
        return self.keyword_chain

    def format_context(self, docs: List[Document]) -> str:
        """Format documents for context"""
        if not docs:
            return f"No relevant {self.insurance_type} insurance policies found."
        
        formatted_parts = []
        for i, doc in enumerate(docs, 1):
            retriever_info = doc.metadata.get('retriever', 'unknown')
            strategy_info = doc.metadata.get('strategy', 'unknown')
            relevance_score = doc.metadata.get('relevance_score', 0)
            
            content = doc.page_content.strip()
            content = re.sub(rf'--- {self.insurance_type.upper()} POLICY (START|END) ---', '', content).strip()
            
            formatted_parts.append(
                f"=== {self.insurance_type.upper()} POLICY OPTION {i} (via {retriever_info}-{strategy_info}, relevance: {relevance_score}) ===\n"
                f"{content}\n"
            )
        
        return "\n".join(formatted_parts)

    def process_document(self, file_path: str, document_id: int) -> bool:
        """Process a document and save chunks to database"""
        try:
            markdown_content = self.load_and_convert_document(file_path)
            policy_chunks, semantic_chunks, header_chunks = self.get_optimized_chunk_strategies(markdown_content)
            self.save_chunks_to_database(document_id, policy_chunks, semantic_chunks, header_chunks)
            logger.info(f"Successfully processed {self.insurance_type} document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Error processing {self.insurance_type} document: {e}")
            return False

    def initialize_system(self, document_id: int = None) -> bool:
        """Initialize the RAG system from database"""
        try:
            self.setup_vector_stores_from_db(document_id)
            self.create_smart_retrievers()
            self.create_rag_chain()
            logger.info(f"{self.insurance_type} RAG system initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Error initializing {self.insurance_type} system: {e}")
            return False

    def query_insurance(self, question: str, save_to_db: bool = True) -> str:
        """Query the insurance system and optionally save to database"""
        if not self.rag_chain:
            raise ValueError(f"{self.insurance_type} RAG system not initialized.")
        
        try:
            start_time = time.time()
            retrieved_docs = self.intelligent_hybrid_retrieve(question, top_k=5)
            chunk_ids = [doc.metadata.get('chunk_id', '') for doc in retrieved_docs]
            response = self.rag_chain.invoke(question)
            processing_time = time.time() - start_time
            
            if save_to_db:
                InsuranceQuery.objects.create(
                    query_text=question,
                    response_text=response,
                    retrieved_chunks=chunk_ids,
                    processing_time=processing_time,
                    insurance_type=self.insurance_type
                )
            
            logger.info(f"{self.insurance_type} query processed in {processing_time:.2f} seconds")
            return response
        except Exception as e:
            logger.error(f"Error processing {self.insurance_type} query: {e}")
            raise

    @staticmethod
    def clear_all_data(insurance_type: str = None):
        """Clear all data from database for a specific insurance type"""
        try:
            if insurance_type:
                DocumentChunk.objects.filter(insurance_type=insurance_type).delete()
                InsuranceDocument.objects.filter(insurance_type=insurance_type).delete()
                InsuranceQuery.objects.filter(insurance_type=insurance_type).delete()
                logger.info(f"All {insurance_type} data cleared from database")
            else:
                DocumentChunk.objects.all().delete()
                InsuranceDocument.objects.all().delete()
                InsuranceQuery.objects.all().delete()
                logger.info("All data cleared from database")
            return True
        except Exception as e:
            logger.error(f"Error clearing {insurance_type or 'all'} data: {e}")
            return False