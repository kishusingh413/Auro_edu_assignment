import numpy as np
from rank_bm25 import BM25Okapi
from .models import Embedding, Document

# Retrieves the top-k most relevant documents using Hybrid Search (BM25 + cosine similarity)
def retrieve_documents(query_text: str, query_embedding, alpha: float = 0.5, top_k: int = 10):
    # Query embeddings as (document_id, embedding)
    embeddings = Embedding.query.with_entities(Embedding.document_id, Embedding.embedding).all()

    if not embeddings:
        return []
    
    # Create a mapping from document_id to embedding
    embedding_dict = {doc_id: embedding for doc_id, embedding in embeddings}

    # Query only the documents that have embeddings, and order by id for consistency.
    doc_ids = list(embedding_dict.keys())
    documents = Document.query.filter(Document.id.in_(doc_ids)).order_by(Document.id).all()

    if not documents:
        return []

    # Build corpus from document content (order must match the documents order)
    corpus = [doc.content for doc in documents]
    tokenized_corpus = [doc.content.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    query_tokens = query_text.lower().split()
    bm25_scores = np.array(bm25.get_scores(query_tokens))

    # Create a NumPy array for embeddings ensuring the order aligns with documents
    all_embeddings = np.array([embedding_dict[doc.id] for doc in documents])
    query_embedding = np.array(query_embedding)

    # Compute cosine similarity scores
    norms = np.linalg.norm(all_embeddings, axis=1) * np.linalg.norm(query_embedding)
    cosine_scores = np.dot(all_embeddings, query_embedding) / (norms + 1e-8)

    # Normalize scores to range [0,1]
    if np.max(bm25_scores) > 0:
        bm25_scores = bm25_scores / np.max(bm25_scores)
    if np.max(cosine_scores) > 0:
        cosine_scores = cosine_scores / np.max(cosine_scores)

    # Combine BM25 and cosine similarity scores using a weighted average
    hybrid_scores = alpha * bm25_scores + (1 - alpha) * cosine_scores

    # Limit top_k to the available number of documents
    k = min(top_k, len(hybrid_scores))
    top_indices = np.argpartition(hybrid_scores, -k)[-k:]
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
