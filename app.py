import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_from_directory
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from flask_cors import CORS


# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load Google API key from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")

# Persistent directory for Chroma vector store
CHROMA_PERSIST_DIR = "chroma_db"

KNOWLEDGE_BASE_FILE = "KnowledgeBase.pdf"

@app.route('/view_default_file')
def view_default_file():
    return send_from_directory(os.getcwd(), KNOWLEDGE_BASE_FILE)

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=None, timeout=None)

# Define prompt template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise. Don't add your own knowledge base and answer using a single line, "
    "and if the question is out of context, tell that you can't answer this question."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Function to process a PDF and update the vector store
def process_pdf(file_path):
    # Load and parse the PDF
    loader = PyPDFLoader(file_path)
    data = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    docs = text_splitter.split_documents(data)

    # Initialize Google Generative AI Embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if the vectorstore already exists; if not, create it with new documents
    if os.path.exists(CHROMA_PERSIST_DIR):
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embeddings
        )
        # Add new documents to the existing vectorstore
        vectorstore.add_documents(docs)
    else:
        # Create a new vectorstore if it doesn't exist
        vectorstore = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_PERSIST_DIR
        )

    # Persist the updated vectorstore to disk
    vectorstore.persist()

    return vectorstore


# Route for chatbot
@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/')
def index():
    return render_template('upload.html')

# Route to handle questions and generate answers using RAG
@app.route('/ask', methods=['POST'])
def ask():
    query = request.json.get("query")
    
    if query:
        # Load the existing Chroma vectorstore from the persistent directory
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        )
        
        # Create a retriever
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

        # Create the retrieval chain (RAG)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        # Get the response from the chain
        response = rag_chain.invoke({"input": query})
        answer = response["answer"] if "answer" in response else "Sorry, I don't have the answer to that."

        return jsonify({"answer": answer})
    
    return jsonify({"answer": "Invalid query."}), 400


# Route to upload a new PDF and process it
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the user opted to use the default file
        if 'use_default_file' in request.form and request.form['use_default_file'] == 'true':
            # Use the default file path
            file_path = os.path.join(os.getcwd(), KNOWLEDGE_BASE_FILE)
            process_pdf(file_path)
            return redirect(url_for('chatbot'))  # Redirect to chatbot after processing

        # Normal file upload handling
        if 'file' not in request.files:
            return jsonify({"error": "No file part."}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file."}), 400
        
        # Save the file to the current working directory
        file_path = os.path.join(os.getcwd(), file.filename)
        file.save(file_path)
        
        # Process the uploaded PDF to update the vector store
        process_pdf(file_path)
        
        return redirect(url_for('chatbot'))  # Redirect to chatbot after upload
    
    return render_template('upload.html')



if __name__ == '__main__':
    app.run(debug=True)