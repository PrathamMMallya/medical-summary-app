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
    """Combine results from multiple retrievers without scoring"""
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
    
    # Return top_k documents without scoring
    return all_docs[:top_k]

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
    
    prompt =  """
You are an intelligent health insurance advisor. Based on the given health condition, age, income, and other personal details, analyze the provided insurance policy data and recommend the most suitable insurance plan(s).

Strictly follow these rules:
- Base your answers ONLY on the context (insurance details).
- Consider whether the health condition (e.g., heart disease) is likely covered under the plan (e.g., critical illness, hospitalization).
- Make sure the premium fits within the person's monthly or yearly budget.
- Recommend at least one plan that matches the user's medical needs and budget. If none are suitable, say so.

Provide your recommendation in bullet points with clear explanation.
Do not hallucinate or fabricate data.

### Input:
{question}

### Context (Insurance Details):
{context}

### Output:"""
    
    # Try different models based on availability
    model_options = [
        "llama3.2:3b",# fallback
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
    return chain, hybrid_retrieval_chain

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
    source = r"C:\Users\tjsre\Desktop\projects\practice\ml\insurance\medical-summary-app\web_scrape_insu.pdf"
    markdown_content = load_and_convert_document(source)
    
    # Create multiple chunking strategies
    header_chunks, small_chunks, large_chunks = get_multiple_chunk_strategies(markdown_content)
    print(f"Chunks created - Header: {len(header_chunks)}, Small: {len(small_chunks)}, Large: {len(large_chunks)}")
    
    # Setup multiple vector stores
    vector_stores = setup_multiple_vector_stores(header_chunks, small_chunks, large_chunks)
    
    # Create multiple retrievers
    retrievers = create_multiple_retrievers(vector_stores)
    
    # Create enhanced RAG chain
    rag_chain, context_chain = create_enhanced_rag_chain(retrievers)
    
    # Test questions
    questions = [
          "Age: 48, Condition: Type 2 Diabetes, Monthly Income: ₹40,000, Needs family floater "    ]
    
    for question in questions:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print(f"{'='*80}")
        
        # Debug retrieval
        debug_hybrid_retrieval(retrievers, question)
        
        # Get context
        context_result = context_chain(question)
        final_context = context_result["context"]
        
        print(f"\n--- FINAL CONTEXT ---")
        print(final_context)
        
        # Get answer
        print(f"\n--- FINAL ANSWER ---")
        try:
            for chunk in rag_chain.stream(question):
                print(chunk, end="", flush=True)
            print("\n")
        except Exception as e:
            print(f"Error generating answer: {e}")