import numpy as np
from rank_bm25 import BM25Okapi
from .models import Embedding, Document

# Retrieves the top-k most relevant documents using Hybrid Search (BM25 + Embeddings)
def retrieve_documents(query_text, query_embedding, alpha=0.5, top_k=10):
    # Fetch documents and embeddings
    documents = Document.query.all()
    embeddings = Embedding.query.with_entities(Embedding.document_id, Embedding.embedding).all()

    if not documents or not embeddings:
        return []

    # Create BM25 index
    corpus = [doc.content for doc in documents]
    tokenized_corpus = [doc.content.lower().split() for doc in documents]  # Tokenization
    bm25 = BM25Okapi(tokenized_corpus)

    # Compute BM25 scores
    query_tokens = query_text.lower().split()
    bm25_scores = np.array(bm25.get_scores(query_tokens))

    # Compute Embedding Similarity (Cosine Similarity)
    doc_ids, all_embeddings = zip(*embeddings)
    all_embeddings = np.array(all_embeddings)
    query_embedding = np.array(query_embedding)

    norms = np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(query_embedding)
    cosine_scores = np.dot(all_embeddings, query_embedding) / (norms + 1e-8)

    # Normalize scores
    if np.max(bm25_scores) > 0:
        bm25_scores = bm25_scores / np.max(bm25_scores)  # Normalize BM25 to [0,1]
    if np.max(cosine_scores) > 0:
        cosine_scores = cosine_scores / np.max(cosine_scores)  # Normalize cosine similarity to [0,1]

    # Compute Hybrid Scores
    hybrid_scores = alpha * bm25_scores + (1 - alpha) * cosine_scores

    # Retrieve top-k documents
    top_indices = np.argpartition(hybrid_scores, -top_k)[-top_k:]
    top_indices = top_indices[np.argsort(hybrid_scores[top_indices])][::-1]

    retrieved_docs = [documents[i] for i in top_indices]
    return retrieved_docs


# def retrieve_documents(query_embedding, top_k=3):
#     embeddings = Embedding.query.all()
#     doc_ids = [emb.document_id for emb in embeddings]
#     all_embeddings = np.array([emb.embedding for emb in embeddings])

#     scores = np.dot(all_embeddings, query_embedding)
#     top_indices = np.argsort(scores)[-top_k:][::-1]

#     retrieved_docs = Document.query.filter(Document.id.in_([doc_ids[i] for i in top_indices])).all()
#     return retrieved_docs
