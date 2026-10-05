from concurrent.futures import ThreadPoolExecutor
from langchain_community.retrievers import ArxivRetriever
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from functools import lru_cache
import threading

cache = {}

# Define the token limit
TOKEN_LIMIT = 200

# Lock for synchronizing file access
file_lock = threading.Lock()

# Function to compute cosine similarity
def cosine_similarity_optimized(key, doc_content):
    """Compute Cosine Similarity between two texts using TF-IDF."""
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([key, doc_content])
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return similarity_matrix[0, 0]  # Return the similarity score

# Function to truncate text to meet token limit
def truncate_text(text, token_limit):
    """Truncate text to a specific token limit."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) > token_limit:
        tokens = tokens[:token_limit]
    return encoding.decode(tokens)

# Function to fetch context for a single key
def fetch_context_for_key(key, max_pages=2, cosine_threshold=0.2):
    """Fetch relevant context for a single key based on cosine similarity."""
    if key in cache:
        return cache[key]

    # Acquire lock before making API calls or accessing resources
    with file_lock:
        retriever_a = ArxivRetriever(load_max_docs=max_pages, get_full_documents=True)
        documents = retriever_a.invoke(key)[:max_pages]  # Use invoke instead of get_relevant_documents

    filtered_context = []

    for doc in documents:
        similarity = cosine_similarity_optimized(key, doc.page_content)
        if similarity >= cosine_threshold:
            filtered_context.append(doc.page_content[:1000])  # Truncate to avoid too-large contexts

    combined_context = "\n\n".join(filtered_context)
    truncated_context = truncate_text(combined_context, TOKEN_LIMIT)

    cache[key] = truncated_context
    return truncated_context

# Function to fetch context for multiple keys in parallel
def fetch_context(keys, max_pages=2, cosine_threshold=0.2):
    """Fetch relevant context for multiple keys in parallel."""
    with ThreadPoolExecutor() as executor:
        results = list(
            executor.map(lambda key: fetch_context_for_key(key, max_pages, cosine_threshold), keys)
        )
    return "\n\n---\n\n".join(results)
