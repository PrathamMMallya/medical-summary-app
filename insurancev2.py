# Environment setup
from dotenv import load_dotenv
import os
import warnings
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama, OllamaEmbeddings
from docling.document_converter import DocumentConverter
from typing import List, Dict, Any
import re

def load_and_convert_document(file_path):
    """Load and convert document to markdown"""
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()

def preprocess_markdown_content(markdown_content):
    """Clean and preprocess markdown content for better chunking"""
    # Remove excessive whitespace
    content = re.sub(r'\n\s*\n\s*\n', '\n\n', markdown_content)
    
    # Ensure policy entries are properly separated
    content = re.sub(r'(\d+\.\s+[A-Za-z]+.*?)\n([A-Z])', r'\1\n\n\2', content)
    
    # Add markers for better chunking
    content = re.sub(r'(\d+\.\s+[A-Za-z][^0-9]*(?:Age|Diseases|Reimbursement|Premium|Highlights).*?)(?=\n\d+\.|\n##|\Z)', 
                     r'--- POLICY START ---\n\1\n--- POLICY END ---', content, flags=re.DOTALL)
    
    return content

def get_optimized_chunk_strategies(markdown_content):
    """Create optimized chunking strategies that preserve policy information"""
    
    # Preprocess content
    processed_content = preprocess_markdown_content(markdown_content)
    
    # Strategy 1: Policy-based chunking (preserve complete policy info)
    policy_chunks = []
    policy_pattern = r'--- POLICY START ---\n(.*?)\n--- POLICY END ---'
    policies = re.findall(policy_pattern, processed_content, re.DOTALL)
    
    for i, policy in enumerate(policies):
        if len(policy.strip()) > 50:  # Only include substantial policies
            from langchain_core.documents import Document
            doc = Document(
                page_content=policy.strip(),
                metadata={'strategy': 'policy', 'policy_id': i}
            )
            policy_chunks.append(doc)
    
    # Strategy 2: Semantic chunks (for better search)
    semantic_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
        separators=["\n--- POLICY END ---", "\n\n", "\n", ". ", " "]
    )
    semantic_chunks = semantic_splitter.create_documents([processed_content])
    
    # Strategy 3: Header-based chunks (for structure)
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    header_chunks = markdown_splitter.split_text(processed_content)
    
    # Add strategy labels
    for chunk in semantic_chunks:
        chunk.metadata['strategy'] = 'semantic'
    for chunk in header_chunks:
        chunk.metadata['strategy'] = 'header'
    
    return policy_chunks, semantic_chunks, header_chunks

def setup_vector_stores(policy_chunks, semantic_chunks, header_chunks):
    """Create optimized vector stores"""
    embeddings = OllamaEmbeddings(model='nomic-embed-text:v1.5', base_url="http://localhost:11434")
    
    stores = {}
    
    # Combine all chunks for a comprehensive store
    all_chunks = policy_chunks + semantic_chunks + header_chunks
    
    # Create main vector store
    single_vector = embeddings.embed_query("health insurance policy")
    index = faiss.IndexFlatL2(len(single_vector))
    main_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    main_store.add_documents(documents=all_chunks)
    stores['main'] = main_store
    
    # Create policy-specific store
    if policy_chunks:
        policy_index = faiss.IndexFlatL2(len(single_vector))
        policy_store = FAISS(
            embedding_function=embeddings,
            index=policy_index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        policy_store.add_documents(documents=policy_chunks)
        stores['policy'] = policy_store
    
    return stores

def create_smart_retrievers(vector_stores):
    """Create smart retrievers with different search strategies"""
    retrievers = {}
    
    # Primary retriever: Similarity search with higher k
    retrievers['primary'] = vector_stores['main'].as_retriever(
        search_type="similarity",
        search_kwargs={'k': 6}
    )
    
    # Secondary retriever: MMR for diversity
    retrievers['diverse'] = vector_stores['main'].as_retriever(
        search_type="mmr",
        search_kwargs={'k': 4, 'fetch_k': 10, 'lambda_mult': 0.3}
    )
    
    # Policy-specific retriever if available
    if 'policy' in vector_stores:
        retrievers['policy'] = vector_stores['policy'].as_retriever(
            search_type="similarity",
            search_kwargs={'k': 5}
        )
    
    return retrievers

def intelligent_hybrid_retrieve(retrievers: Dict, question: str, top_k: int = 5):
    """Intelligent hybrid retrieval with deduplication and ranking"""
    all_docs = []
    seen_content = set()
    
    # Extract key terms from question for better matching
    question_lower = question.lower()
    key_terms = []
    
    # Extract age
    age_match = re.search(r'age[:\s]*(\d+)', question_lower)
    if age_match:
        key_terms.append(f"age {age_match.group(1)}")
    
    # Extract conditions
    conditions = ['diabetes', 'heart', 'cancer', 'hypertension', 'kidney', 'liver']
    for condition in conditions:
        if condition in question_lower:
            key_terms.append(condition)
    
    # Extract budget info
    budget_match = re.search(r'₹(\d+,?\d*)', question)
    if budget_match:
        key_terms.append(f"budget {budget_match.group(1)}")
    
    print(f"Extracted key terms: {key_terms}")
    
    # Get documents from each retriever
    for name, retriever in retrievers.items():
        try:
            docs = retriever.invoke(question)
            for doc in docs:
                # Create a more sophisticated content hash
                content_signature = ' '.join(doc.page_content.split()[:20])  # First 20 words
                content_hash = hash(content_signature)
                
                if content_hash not in seen_content:
                    # Add relevance score based on key terms
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
            print(f"Error with {name} retriever: {e}")
    
    # Sort by relevance score and return top_k
    all_docs.sort(key=lambda x: x.metadata.get('relevance_score', 0), reverse=True)
    return all_docs[:top_k]

def format_context_intelligently(docs):
    """Format documents with better structure"""
    if not docs:
        return "No relevant insurance policies found."
    
    formatted_parts = []
    
    for i, doc in enumerate(docs, 1):
        retriever_info = doc.metadata.get('retriever', 'unknown')
        strategy_info = doc.metadata.get('strategy', 'unknown')
        relevance_score = doc.metadata.get('relevance_score', 0)
        
        # Clean up the content
        content = doc.page_content.strip()
        content = re.sub(r'--- POLICY (START|END) ---', '', content).strip()
        
        formatted_parts.append(
            f"=== POLICY OPTION {i} (via {retriever_info}-{strategy_info}, relevance: {relevance_score}) ===\n"
            f"{content}\n"
        )
    
    return "\n".join(formatted_parts)

def create_enhanced_rag_chain(retrievers):
    """Create an enhanced RAG chain with better prompting"""
    
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

Base your recommendations ONLY on the provided policy information. Do not make assumptions about policies not mentioned in the context.
"""
    
    # Try to use the best available model
    model_options = ["llama3.2:3b", "llama3.1:8b", "llama3:8b"]
    
    model = None
    for model_name in model_options:
        try:
            model = ChatOllama(
                model=model_name, 
                base_url="http://localhost:11434",
                temperature=0.1,  # Lower temperature for more consistent responses
                num_predict=1000  # Ensure longer responses
            )
            print(f"Successfully loaded model: {model_name}")
            break
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            continue
    
    if not model:
        raise Exception("No suitable model found. Please ensure Ollama is running with a compatible model.")
    
    prompt_template = ChatPromptTemplate.from_template(prompt)
    
    def enhanced_retrieval_chain(question):
        docs = intelligent_hybrid_retrieve(retrievers, question, top_k=5)
        context = format_context_intelligently(docs)
        return {"context": context, "question": question}
    
    chain = (
        enhanced_retrieval_chain
        | prompt_template
        | model
        | StrOutputParser()
    )
    
    return chain, enhanced_retrieval_chain

def debug_retrieval_process(retrievers, question):
    """Enhanced debug function"""
    print(f"\n{'='*80}")
    print(f"DEBUGGING RETRIEVAL FOR: {question}")
    print(f"{'='*80}")
    
    # Show individual retriever results
    for name, retriever in retrievers.items():
        print(f"\n--- {name.upper()} RETRIEVER RESULTS ---")
        try:
            docs = retriever.invoke(question)
            for i, doc in enumerate(docs[:3]):  # Show top 3
                content_preview = doc.page_content[:300].replace('\n', ' ')
                print(f"{i+1}. {content_preview}...")
                
                # Highlight if contains relevant terms
                doc_lower = doc.page_content.lower()
                if any(term in doc_lower for term in ['diabetes', 'premium', 'age', '₹']):
                    print("   *** CONTAINS RELEVANT TERMS ***")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Show final hybrid results
    print(f"\n--- FINAL HYBRID RETRIEVAL RESULTS ---")
    final_docs = intelligent_hybrid_retrieve(retrievers, question)
    for i, doc in enumerate(final_docs):
        relevance = doc.metadata.get('relevance_score', 0)
        content_preview = doc.page_content[:200].replace('\n', ' ')
        print(f"Final-{i+1} (relevance: {relevance}): {content_preview}...")

# Main execution
if __name__ == "__main__":
    try:
        # Load document
        source = r"C:\Users\tjsre\Desktop\projects\practice\ml\insurance\medical-summary-app\web_scrape_insu.pdf"
        
        print("Loading and converting document...")
        markdown_content = load_and_convert_document(source)
        
        print("Creating optimized chunking strategies...")
        policy_chunks, semantic_chunks, header_chunks = get_optimized_chunk_strategies(markdown_content)
        print(f"Chunks created - Policy: {len(policy_chunks)}, Semantic: {len(semantic_chunks)}, Header: {len(header_chunks)}")
        
        print("Setting up vector stores...")
        vector_stores = setup_vector_stores(policy_chunks, semantic_chunks, header_chunks)
        
        print("Creating smart retrievers...")
        retrievers = create_smart_retrievers(vector_stores)
        
        print("Creating enhanced RAG chain...")
        rag_chain, context_chain = create_enhanced_rag_chain(retrievers)
        
        # Test question
        test_question = "my age is 30, I have diabetes and hypertension, I want a health insurance policy that covers these conditions with a premium of ₹50000 per year. What are my options?"
        
        print(f"\n{'='*80}")
        print(f"PROCESSING QUESTION: {test_question}")
        print(f"{'='*80}")
        
        # Debug retrieval process
        debug_retrieval_process(retrievers, test_question)
        
        # Get and display context
        print(f"\n--- RETRIEVED CONTEXT ---")
        context_result = context_chain(test_question)
        print(context_result["context"])
        
        # Generate and display answer
        print(f"\n--- GENERATED RECOMMENDATION ---")
        try:
            response = ""
            for chunk in rag_chain.stream(test_question):
                print(chunk, end="", flush=True)
                response += chunk
            print(f"\n\n{'='*80}")
            print("PROCESSING COMPLETE")
            print(f"{'='*80}")
            
        except Exception as e:
            print(f"Error generating response: {e}")
            print("Trying alternative approach...")
            
            # Fallback: get response without streaming
            try:
                response = rag_chain.invoke(test_question)
                print(response)
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                
    except Exception as e:
        print(f"Critical error in main execution: {e}")
        import traceback
        traceback.print_exc()