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

from langchain_text_splitters import MarkdownHeaderTextSplitter

from langchain_ollama import ChatOllama, OllamaEmbeddings

from docling.document_converter import DocumentConverter
def load_and_convert_document(file_path):
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()

def get_markdown_splits(markdown_content):
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    return markdown_splitter.split_text(markdown_content)

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
    return "\n\n".join([doc.page_content for doc in docs])

# Setting up the RAG chain
def create_rag_chain(retriever):
    prompt = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
        ### Question: {question} 
        
        ### Context: {context} 
        
        ### Answer:
    """
    model = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")
    prompt_template = ChatPromptTemplate.from_template(prompt)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )
    return chain
# One-time process



# Load document
source =  r"C:\Users\tjsre\Downloads\90ade7e39d5e481f9aeb772a19a30234.pdf"
markdown_content = load_and_convert_document(source)
chunks = get_markdown_splits(markdown_content)

# Create vector store
vector_store = setup_vector_store(chunks)

# Setup retriever
retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})

# Create RAG chain
rag_chain = create_rag_chain(retriever)
question = "what would be restricted to a maximum of INR in accidental hospitalization"

print(f"Question: {question}")
for chunk in rag_chain.stream(question):
    print(chunk, end="", flush=True)
print("\n" + "-" * 50 + "\n")