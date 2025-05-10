from flask import Flask, render_template, jsonify, request
import os
from werkzeug.utils import secure_filename
from src.helper import load_pdf_directory, text_split, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_together import ChatTogether
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain.document_loaders import PyPDFLoader
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import uuid

app = Flask(__name__)  
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    # Check if any file part exists
    if 'files[]' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    files = request.files.getlist('files[]')
    
    # Check if any file was selected
    if not files or files[0].filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Process valid files
    uploaded_files = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            uploaded_files.append(file_path)
    
    if not uploaded_files:
        return jsonify({'error': 'No valid files uploaded'}), 400
    
    # Process uploaded PDFs
    session_id = str(uuid.uuid4())
    try:
        process_uploaded_pdfs(uploaded_files, session_id)
        # Delete files after processing
        for file_path in uploaded_files:
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")
        return jsonify({'success': True, 'session_id': session_id}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_uploaded_pdfs(file_paths, session_id):
    # Load PDFs
    documents = []

    documents = load_pdf_directory('uploads/')
    
    # Split documents into chunks
    text_chunks = text_split(documents)
    
    # Get embeddings
    embeddings = download_hugging_face_embeddings()

    index_name = "pdfbot"  # You might want to make this dynamic per session

    # Check if index exists, create if it doesn't
    try:
        existing_indexes = pc.list_indexes()
        existing_index_names = [idx["name"] for idx in existing_indexes]

        if index_name not in existing_index_names:
            pc.create_index(
                name=index_name,
                dimension=384,  # Dimension for sentence-transformers/all-MiniLM-L6-v2
                metric="cosine",
                spec=ServerlessSpec(
                    cloud = "aws",
                    region = "us-east-1"
                )
            )
            print(f" Created index '{index_name}'")
        else:
            print(f"Index '{index_name}' already exists")
    except Exception as e:
        print(f" Error checking or creating index: {e}")

    
    # Create vector store
    docsearch = PineconeVectorStore.from_documents(
        documents=text_chunks,
        index_name=index_name,
        embedding=embeddings,
        namespace=session_id  # Use session_id as namespace to isolate user data
    )
    
    return docsearch

@app.route('/ask', methods=["POST"])
def chat():
    msg = request.form.get('msg')
    session_id = request.form.get('session_id')
    
    if not msg or not session_id:
        return "Please provide a message and session ID", 400
    
    # Get embeddings
    embeddings = download_hugging_face_embeddings()
    
    # Connect to Pinecone
    index_name = "pdfbot"
    
    # Create vector store
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
        namespace=session_id  # Use session_id to isolate user data
    )
    
    # Create retriever
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    # Initialize the Together AI chat model
    llm = ChatTogether(
        model="mistralai/Mistral-7B-Instruct-v0.1",  
        temperature=0.4,
        max_tokens=500
    )
    
    # Create prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    # Create QA chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # Get response
    response = rag_chain.invoke({"input": msg})
    
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
