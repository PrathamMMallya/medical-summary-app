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

def load_and_convert_document(file_path):
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()

def get_improved_splits(markdown_content):
    # First split by headers
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    header_splits = markdown_splitter.split_text(markdown_content)
    
    # Then use recursive splitter for better chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased chunk size
        chunk_overlap=200,  # Increased overlap
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Apply recursive splitting to each header split
    final_chunks = []
    for chunk in header_splits:
        sub_chunks = text_splitter.split_documents([chunk])
        final_chunks.extend(sub_chunks)
    
    return final_chunks

def setup_vector_store(chunks):
    embeddings = OllamaEmbeddings(model='nomic-embed-text:v1.5', base_url="http://localhost:11434")
    single_vector = embeddings.embed_query("this is some text data")
    index = faiss.IndexFlatL2(len(single_vector))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(documents=chunks)
    return vector_store

def format_docs(docs):
    return "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])

def create_rag_chain(retriever):
    prompt = """
    You are an assistant for question-answering tasks about insurance policies. 
    Use the following pieces of retrieved context to answer the question.
    
    IMPORTANT: Look carefully for specific amounts, limits, and numerical values in the context.
    If you find specific INR amounts or monetary limits, include them in your answer.
    If you don't know the answer, just say that you don't know.
    Answer in bullet points and be specific about numerical values.
    
    ### Question: {question}
    
    ### Context: 
    {context}
    
    ### Answer:
    """
    
    model = ChatOllama(model="llama3.2:3b", base_url="http://localhost:11434")  # Alternative 3B model
    prompt_template = ChatPromptTemplate.from_template(prompt)
    
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )
    return chain

def debug_retrieval(retriever, question):
    """Debug function to see what documents are being retrieved"""
    docs = retriever.get_relevant_documents(question)
    print("=== RETRIEVED DOCUMENTS ===")
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(f"Content: {doc.page_content[:500]}...")
        print(f"Metadata: {doc.metadata}")
    print("=" * 50)
    return docs

# One-time process
source = r"C:\Users\tjsre\Downloads\90ade7e39d5e481f9aeb772a19a30234.pdf"
markdown_content = load_and_convert_document(source)

# Use improved chunking
chunks = get_improved_splits(markdown_content)
print(f"Total chunks created: {len(chunks)}")

# Create vector store
vector_store = setup_vector_store(chunks)

# Setup retriever with better parameters
retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={
        'k': 5,  # Increased from 3 to 5
        'fetch_k': 10,  # Fetch more candidates
        'lambda_mult': 0.7  # Balance between relevance and diversity
    }
)

# Create RAG chain
rag_chain = create_rag_chain(retriever)

# Test questions
questions = [
    "what would be restricted to a maximum of INR in accidental hospitalization",
]

for question in questions:
    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")
    
    # Debug: Show retrieved documents
    debug_retrieval(retriever, question)
    
    # Get answer
    print("\nAnswer:")
    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
    print("\n")

# Additional debugging: Search for specific terms
print("\n" + "="*60)
print("SEARCHING FOR SPECIFIC TERMS")
print("="*60)

search_terms = ["INR 1,00,000", "accidental hospitalisation", "125%", "maximum"]
for term in search_terms:
    docs = retriever.get_relevant_documents(term)
    print(f"\nSearch term: '{term}' - Found {len(docs)} documents")
    for i, doc in enumerate(docs[:2]):  # Show first 2 matches
        if "1,00,000" in doc.page_content or "125%" in doc.page_content:
            print(f"*** RELEVANT MATCH FOUND ***")
            print(f"Content: {doc.page_content}")
            print("-" * 40)