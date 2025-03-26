from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from .models import db, User, Document, Embedding, SelectedDocument
from .embeddings import generate_embedding
from .retrieval import retrieve_documents

main = Blueprint("main", __name__)

# Document Ingestion
@main.route("/upload", methods=["POST"])
@jwt_required()
def upload_document():
    data = request.json
    if "title" not in data or "content" not in data:
        return jsonify({"error": "Title and content are required"}), 400

    document = Document(title=data["title"], content=data["content"])
    db.session.add(document)
    db.session.commit()

    # Generate embeddings
    embedding_vector = generate_embedding(data["content"])
    embedding_entry = Embedding(document_id=document.id, embedding=embedding_vector)
    db.session.add(embedding_entry)
    db.session.commit()

    return jsonify({"message": "Document stored successfully"}), 201

# Document Selection API
@main.route("/select-documents", methods=["POST"])
@jwt_required()
def select_documents():
    user_id = get_jwt_identity()
    data = request.json
    if "document_ids" not in data:
        return jsonify({"error": "Document IDs are required"}), 400

    # Validate document existence
    selected_docs = Document.query.filter(Document.id.in_(data["document_ids"])).all()
    if not selected_docs:
        return jsonify({"error": "No valid documents found"}), 404

    # Store the selected documents in the database for the user
    SelectedDocument.query.filter_by(user_id=user_id).delete()  # Clear previous selections
    for doc in selected_docs:
        selection = SelectedDocument(user_id=user_id, document_id=doc.id)
        db.session.add(selection)
    
    db.session.commit()
    return jsonify({"selected_documents": [doc.id for doc in selected_docs]}), 200

# Q&A Retrieval
@main.route("/ask", methods=["POST"])
@jwt_required()
def ask_question():
    user_id = get_jwt_identity()
    data = request.json
    if "question" not in data:
        return jsonify({"error": "Question is required"}), 400

    # Retrieve user's selected documents
    selected_doc_ids = [sd.document_id for sd in SelectedDocument.query.filter_by(user_id=user_id).all()]
    if not selected_doc_ids:
        return jsonify({"error": "No documents selected. Please select documents first."}), 400

    # Generate embedding and retrieve docs
    question_embedding = generate_embedding(data["question"])
    retrieved_docs = retrieve_documents(question_embedding)

    # Filter retrieved documents based on selected ones
    filtered_docs = [doc for doc in retrieved_docs if doc.id in selected_doc_ids]

    if not filtered_docs:
        return jsonify({"error": "No relevant document found in selected documents"}), 404

    answers = [{"title": doc.title, "content": doc.content[:300]} for doc in filtered_docs]
    return jsonify(answers=answers), 200