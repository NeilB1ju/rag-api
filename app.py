from flask import Flask, request, jsonify
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
import os
import spacy

app = Flask(__name__)

# Initialize global variables
chat_history = []
cached_llm = Ollama(model="llama3")
nlp = spacy.load("en_core_web_trf")
print("Initializing FastEmbedEmbeddings...")
embedding = FastEmbedEmbeddings()
if embedding is None:
    print("Embedding model initialization failed.")
else:
    print("Embedding model initialized successfully.")

# Ensure the database directory exists
db_directory = "db"
os.makedirs(db_directory, exist_ok=True)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=80, length_function=len, is_separator_regex=False
)

raw_prompt = PromptTemplate.from_template(
    """
    <s>[INST] You are a technical assistant good at searching documents. If you do not have an answer from the provided information say so. [/INST] </s>
    [INST] {input}
           Context: {context}
           Answer:
    [/INST]
"""
)

def get_vector_store(pdf_id):
    """Helper function to get the vector store for a specific PDF."""
    pdf_vector_store_path = os.path.join(db_directory, pdf_id)  # No .pdf extension here
    print(f"Looking for vector store at: {pdf_vector_store_path}")

    if not os.path.exists(pdf_vector_store_path):
        print("Vector store not found.")
        return None
    
    return Chroma(persist_directory=pdf_vector_store_path, embedding_function=embedding)

@app.route("/ask_pdf/<pdf_id>", methods=["POST"])
def ask_pdf_post(pdf_id):
    json_content = request.json
    print(f"Received request for PDF ID: {pdf_id}")
    print(f"Request content: {json_content}")  # Log request content for debugging

    query = json_content.get("query")
    
    if query is None:
        return jsonify({"error": "The 'query' field is required"}), 400  # Return error if 'query' is missing
    
    vector_store = get_vector_store(pdf_id)
    if vector_store is None:
        return jsonify({"error": "PDF vector store not found"}), 404

    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.2},
    )

    retriever_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            (
                "human",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation",
            ),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm=cached_llm, retriever=retriever, prompt=retriever_prompt
    )

    document_chain = create_stuff_documents_chain(cached_llm, raw_prompt)
    retrieval_chain = create_retrieval_chain(history_aware_retriever, document_chain)

    result = retrieval_chain.invoke({"input": query})

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result["answer"]))

    sources = [{"source": doc.metadata["source"], "page_content": doc.page_content} for doc in result["context"]]

    return jsonify({"answer": result["answer"], "sources": sources})

@app.route("/pdf", methods=["POST"])
def pdf_post():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_name = file.filename
    pdf_vector_store_path = os.path.join(db_directory, file_name)  # Use file name without .pdf

    # Save the PDF file
    save_file = os.path.join("pdf", file_name)
    os.makedirs("pdf", exist_ok=True)  # Ensure the pdf directory exists
    file.save(save_file)

    # Load and process the PDF
    loader = PDFPlumberLoader(save_file)
    docs = loader.load_and_split()
    chunks = text_splitter.split_documents(docs)

    # Create the vector store
    try:
        vector_store = Chroma.from_documents(
            documents=chunks, embedding=embedding, persist_directory=pdf_vector_store_path
        )
        vector_store.persist()
        print(f"Vector store created and persisted at: {pdf_vector_store_path}")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return jsonify({"error": "vector store not created: " + str(e)}), 500

    return jsonify({
        "status": "Successfully Uploaded",
        "filename": file_name,
        "doc_len": len(docs),
        "chunks": len(chunks),
    })

@app.route("/summarize_pdf/<pdf_id>", methods=["GET"])
def summarize_pdf(pdf_id):
    vector_store = get_vector_store(pdf_id)
    if vector_store is None:
        return jsonify({"error": "PDF vector store not found"}), 404

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 100})
    docs = retriever.get_relevant_documents("")

    document_contents = "\n\n".join([doc.page_content for doc in docs])
    
    summary_prompt = PromptTemplate.from_template(
        """
        <s>[INST] You are a summarization assistant. Please summarize the following documents. [/INST] </s>
        {documents}
        Answer:
        [/INST]
        """
    )

    summary_input = summary_prompt.format(documents=document_contents)
    summary_response = cached_llm.invoke(summary_input)

    return jsonify({"summary": summary_response})

@app.route("/extract_entities/<pdf_id>", methods=["GET"])
def extract_entities(pdf_id):
    vector_store = get_vector_store(pdf_id)
    if vector_store is None:
        return jsonify({"error": "PDF vector store not found"}), 404

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 1})
    docs = retriever.get_relevant_documents("")
    
    document_contents = "\n\n".join([doc.page_content for doc in docs])
    doc = nlp(document_contents)

    allowed_entity_types = ["ORG", "PERSON", "GPE", "LOC"]
    entities = [{"text": ent.text, "label": ent.label_} for ent in doc.ents if ent.label_ in allowed_entity_types and len(ent.text) > 1]

    return jsonify({"entities": entities})

def start_app():
    app.run(host="0.0.0.0", port=8080, debug=True)

if __name__ == "__main__":
    start_app()
