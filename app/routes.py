from flask import Blueprint, request, jsonify, current_app
from flask_jwt_extended import jwt_required, get_jwt_identity
from werkzeug.utils import secure_filename
import PyPDF2
import os

from .models import db, Document, Embedding, SelectedDocument
from .embeddings import generate_embedding
from .retrieval import retrieve_documents

main = Blueprint("main", __name__)

# Upload a new document
@main.route("/upload", methods=["POST"])
@jwt_required()
def upload_document():
    # Retrieve user id from JWT identity
    user_id = get_jwt_identity()

    # Check if file is provided and valid
    if "file" not in request.files or not request.files["file"].filename:
        return jsonify({"error": "No selected file"}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)

    # Enforce file extension to be .pdf
    if not filename.lower().endswith(".pdf"):
        return jsonify({"error": "Only PDF files are allowed"}), 400

    upload_folder = current_app.config.get("UPLOAD_FOLDER")
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    file_path = os.path.join(upload_folder, filename)
    
    try:
        file.save(file_path)
    except Exception as e:
        return jsonify({"error": f"File saving failed: {str(e)}"}), 500

    # Extract text content from PDF using PyPDF2
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            content = ""
            for page in reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    content += extracted_text
    except Exception as e:
        return jsonify({"error": f"Failed to read file: {str(e)}"}), 500

    title = request.form.get("title", filename)
    document = Document(user_id=user_id, title=title, content=content, file_path=file_path)
    db.session.add(document)
    db.session.commit()

    # Generate embeddings for the document content
    try:
        embedding_vector = generate_embedding(document.content)
    except Exception as e:
        current_app.logger.error(f"Embedding generation failed: {e}")
        return jsonify({"error": f"Embedding generation failed: {str(e)}"}), 500

    embedding_entry = Embedding(document_id=document.id, embedding=embedding_vector.tolist())
    db.session.add(embedding_entry)
    db.session.commit()

    return jsonify({"message": "Document stored successfully", "document_id": document.id}), 201

# Retrieve all the user documents
@main.route("/documents", methods=["GET"])
@jwt_required()
def get_documents():
    user_id = get_jwt_identity()
    documents = Document.query.filter_by(user_id=user_id).all()
    return jsonify([{"id": doc.id, "title": doc.title} for doc in documents]), 200

# Select documents considered for Q&A retrieval
@main.route("/select-documents", methods=["POST"])
@jwt_required()
def select_documents():
    user_id = get_jwt_identity()
    data = request.get_json()
    if not data or "document_ids" not in data:
        return jsonify({"error": "Document IDs are required"}), 400

    # Validate document existence
    selected_docs = Document.query.filter(Document.id.in_(data["document_ids"])).all()
    if not selected_docs:
        return jsonify({"error": "No valid documents found"}), 404

    # Clear previous selections for this user
    SelectedDocument.query.filter_by(user_id=user_id).delete()
    for doc in selected_docs:
        selection = SelectedDocument(user_id=user_id, document_id=doc.id)
        db.session.add(selection)
    
    db.session.commit()
    return jsonify({"selected_documents": [doc.id for doc in selected_docs]}), 200

# Accepts a user question, generates an embedding, retrieves relevant documents, and returns a summary of document content
@main.route("/ask", methods=["POST"])
@jwt_required()
def ask_question():
    user_id = get_jwt_identity()
    data = request.get_json()
    if not data or "question" not in data:
        return jsonify({"error": "Question is required"}), 400

    # Retrieve user's selected documents
    selected_doc_ids = [sd.document_id for sd in SelectedDocument.query.filter_by(user_id=user_id).all()]
    if not selected_doc_ids:
        return jsonify({"error": "No documents selected. Please select documents first."}), 400

    # Generate the question embedding
    try:
        question_embedding = generate_embedding(data["question"])
    except Exception as e:
        current_app.logger.error(f"Question embedding generation failed: {e}")
        return jsonify({"error": f"Embedding generation failed: {str(e)}"}), 500

    # Retrieve relevant documents based on the question embedding
    retrieved_docs = retrieve_documents(data["question"], question_embedding)

    # Filter retrieved documents based on selected ones
    filtered_docs = [doc for doc in retrieved_docs if doc.id in selected_doc_ids]

    if not filtered_docs:
        return jsonify({"error": "No relevant document found in selected documents"}), 404

    answers = [{"title": doc.title, "content": doc.content[:300]} for doc in filtered_docs]
    return jsonify(answers=answers), 200