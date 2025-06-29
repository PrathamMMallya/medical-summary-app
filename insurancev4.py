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
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()

def get_multiple_chunk_strategies(markdown_content):
    """Create multiple chunking strategies for better coverage"""
    
    # Strategy 1: Header-based chunking (for structure)
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    header_chunks = markdown_splitter.split_text(markdown_content)
    
    # Strategy 2: Small overlapping chunks (for precision)
    small_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    small_chunks = small_splitter.create_documents([markdown_content])
    
    # Strategy 3: Large chunks (for context)
    large_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    large_chunks = large_splitter.create_documents([markdown_content])
    
    # Add strategy labels
    for chunk in header_chunks:
        chunk.metadata['strategy'] = 'header'
    for chunk in small_chunks:
        chunk.metadata['strategy'] = 'small'
    for chunk in large_chunks:
        chunk.metadata['strategy'] = 'large'
    
    return header_chunks, small_chunks, large_chunks

def setup_multiple_vector_stores(header_chunks, small_chunks, large_chunks):
    """Create multiple vector stores with different strategies"""
    embeddings = OllamaEmbeddings(model='nomic-embed-text:v1.5', base_url="http://localhost:11434")
    
    stores = {}
    
    for name, chunks in [('header', header_chunks), ('small', small_chunks), ('large', large_chunks)]:
        single_vector = embeddings.embed_query("insurance policy document")
        index = faiss.IndexFlatL2(len(single_vector))
        vector_store = FAISS(
            embedding_function=embeddings,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )
        vector_store.add_documents(documents=chunks)
        stores[name] = vector_store
    
    return stores

def create_multiple_retrievers(vector_stores):
    """Create retrievers with different search strategies"""
    retrievers = {}
    
    # Retriever 1: Similarity search (precise)
    retrievers['similarity'] = vector_stores['small'].as_retriever(
        search_type="similarity",
        search_kwargs={'k': 3}
    )
    
    # Retriever 2: MMR search (diverse)
    retrievers['mmr'] = vector_stores['header'].as_retriever(
        search_type="mmr",
        search_kwargs={'k': 4, 'fetch_k': 8, 'lambda_mult': 0.5}
    )
    
    # Retriever 3: Large context search
    retrievers['context'] = vector_stores['large'].as_retriever(
        search_type="similarity",
        search_kwargs={'k': 2}
    )
    
    return retrievers

def hybrid_retrieve(retrievers: Dict, question: str, top_k: int = 5):
    """Combine results from multiple retrievers"""
    all_docs = []
    seen_content = set()
    
    # Get documents from each retriever
    for name, retriever in retrievers.items():
        try:
            docs = retriever.invoke(question)
            for doc in docs:
                # Avoid duplicates based on content similarity
                content_hash = hash(doc.page_content[:200])  # Use first 200 chars as hash
                if content_hash not in seen_content:
                    doc.metadata['retriever'] = name
                    all_docs.append(doc)
                    seen_content.add(content_hash)
        except Exception as e:
            print(f"Error with {name} retriever: {e}")
    
    # Score and rank documents
    scored_docs = []
    for doc in all_docs:
        score = calculate_relevance_score(doc.page_content, question)
        scored_docs.append((score, doc))
    
    # Sort by score and return top_k
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs[:top_k]]

def calculate_relevance_score(content: str, question: str) -> float:
    """Simple relevance scoring based on keyword matching"""
    question_lower = question.lower()
    content_lower = content.lower()
    
    # Key terms for insurance questions
    key_terms = [
        'inr', 'rupees', 'amount', 'limit', 'maximum', 'minimum',
        'accidental', 'hospitalization', 'hospitalisation', 'sum insured',
        '1,00,000', '125%', 'restricted', 'benefit'
    ]
    
    score = 0
    
    # Exact phrase matching
    if 'accidental hospitalization' in question_lower or 'accidental hospitalisation' in question_lower:
        if 'accidental' in content_lower and 'hospital' in content_lower:
            score += 3
    
    # Numerical value bonus
    if re.search(r'inr[\s]*[\d,]+', content_lower):
        score += 2
    
    # Key term matching
    for term in key_terms:
        if term in content_lower:
            score += 1
    
    # Length penalty for very short content
    if len(content) < 100:
        score -= 1
    
    return score

def format_hybrid_docs(docs):
    """Format documents with retriever information"""
    formatted = []
    for i, doc in enumerate(docs):
        retriever_info = doc.metadata.get('retriever', 'unknown')
        strategy_info = doc.metadata.get('strategy', 'unknown')
        formatted.append(f"Document {i+1} (via {retriever_info}-{strategy_info}):\n{doc.page_content}")
    return "\n\n".join(formatted)

def create_enhanced_rag_chain(retrievers):
    """Create RAG chain with hybrid retrieval"""
    
    prompt = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
        ### Question: {question} 
        
        ### Context: {context} 
        
        ### Answer:"""
    
    # Try different models based on availability
    model_options = [
        "deepseek-r1:1.5b",  # fallback
    ]
    
    model = None
    for model_name in model_options:
        try:
            model = ChatOllama(model=model_name, base_url="http://localhost:11434")
            print(f"Using model: {model_name}")
            break
        except:
            continue
    
    if not model:
        raise Exception("No suitable model found")
    
    prompt_template = ChatPromptTemplate.from_template(prompt)
    
    def hybrid_retrieval_chain(question):
        docs = hybrid_retrieve(retrievers, question, top_k=5)
        context = format_hybrid_docs(docs)
        return {"context": context, "question": question}
    
    chain = (
        hybrid_retrieval_chain
        | prompt_template
        | model
        | StrOutputParser()
    )
    return chain

def debug_hybrid_retrieval(retrievers, question):
    """Debug the hybrid retrieval process"""
    print(f"\n=== HYBRID RETRIEVAL DEBUG for: {question} ===")
    
    for name, retriever in retrievers.items():
        print(f"\n--- {name.upper()} RETRIEVER ---")
        try:
            docs = retriever.invoke(question)
            for i, doc in enumerate(docs[:2]):  # Show top 2 from each
                print(f"{name}-{i+1}: {doc.page_content[:200]}...")
                if 'inr' in doc.page_content.lower() or '1,00,000' in doc.page_content:
                    print("*** CONTAINS INR AMOUNT ***")
        except Exception as e:
            print(f"Error: {e}")
    
    # Show final hybrid results
    print(f"\n--- FINAL HYBRID RESULTS ---")
    final_docs = hybrid_retrieve(retrievers, question)
    for i, doc in enumerate(final_docs):
        print(f"Final-{i+1}: {doc.page_content[:200]}...")
        if 'inr' in doc.page_content.lower() or '1,00,000' in doc.page_content:
            print("*** CONTAINS INR AMOUNT ***")

# Main execution
if __name__ == "__main__":
    # Load document
    source = r"C:\Users\tjsre\Downloads\90ade7e39d5e481f9aeb772a19a30234.pdf"
    markdown_content = load_and_convert_document(source)
    
    # Create multiple chunking strategies
    header_chunks, small_chunks, large_chunks = get_multiple_chunk_strategies(markdown_content)
    print(f"Chunks created - Header: {len(header_chunks)}, Small: {len(small_chunks)}, Large: {len(large_chunks)}")
    
    # Setup multiple vector stores
    vector_stores = setup_multiple_vector_stores(header_chunks, small_chunks, large_chunks)
    
    # Create multiple retrievers
    retrievers = create_multiple_retrievers(vector_stores)
    
    # Create enhanced RAG chain
    rag_chain = create_enhanced_rag_chain(retrievers)
    
    # Test questions
    questions = [
        "who is sreeharish TJ",
    ]
    
    for question in questions:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}")
        
        # Debug retrieval
        debug_hybrid_retrieval(retrievers, question)
        
        # Get answer
        print(f"\n--- FINAL ANSWER ---")
        try:
            for chunk in rag_chain.stream(question):
                print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"Error generating answer: {e}")