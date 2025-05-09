from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader


# Load PDFs from a directory
def load_pdf_directory(data_dir):
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    return documents

# Load a single PDF file
def load_pdf_file(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# Split the text into smaller chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Download the embeddings from huggingface
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddings
