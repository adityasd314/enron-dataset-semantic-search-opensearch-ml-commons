import argparse
import json
from opensearchpy import OpenSearch, RequestsHttpConnection
from sentence_transformers import SentenceTransformer
import os
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()
# --- Configuration ---
OPENSEARCH_HOST = os.getenv('OPENSEARCH_HOST', 'localhost')
OPENSEARCH_PORT = os.getenv('OPENSEARCH_PORT', 9200)
OPENSEARCH_USER = os.getenv('OPENSEARCH_USER', 'admin')
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD', '') # set Environment variable or set second argument as password
INDEX_NAME = os.getenv('INDEX_NAME', 'my-email-data')  # The OpenSearch index name you created
EMBEDDINGS_DIMENSION = 384
model = SentenceTransformer(EMBEDDING_MODEL_NAME) # only for converting the query into a embedding

# --- OpenSearch Client Setup ---
client = OpenSearch(
    hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
    http_compress=True,
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
    use_ssl=True,
    verify_certs=False,  # False for self-signed certificates, True for valid CA-signed certs
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    connection_class=RequestsHttpConnection
)

# --- Functions ---
def load_json_data(file_path):
    """returns a dictionary with uid as key"""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                uid = record.get("uid")
                if uid:
                    data[uid] = record
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data

def generate_query_embedding(query_text):
    """Generates an embedding for the search query using the local model."""
    if not query_text:
        return None
    embeddings = model.encode([query_text])
    return embeddings[0].tolist()

def perform_knn_search(query_embedding, target_field, k=5):
    """
    Performs a k-NN search in OpenSearch.

    Args:
        query_embedding (list): The embedding vector of the search query.
        target_field (str): The name of the knn_vector field to search against (e.g., 'subject_embedding', 'body_embedding').
        k (int): The number of nearest neighbors to retrieve.
    """
    search_body = {
        "size": k,
        "query": {
            "knn": {
                target_field: {
                    "vector": query_embedding,
                    "k": k
                }
            }
        }
    }

    print(f"\nSearching for query in '{target_field}'...")
    try:
        response = client.search(
            index=INDEX_NAME,
            body=search_body
        )
        return response
    except Exception as e:
        print(f"Error during k-NN search: {e}")
        return None

def print_search_results(response):
    """Prints the search results in a readable format."""
    data_json = load_json_data("enron_emails_sample.json")
    if response and response['hits']['hits']:
        print(f"Found {response['hits']['total']['value']} hits:\n")
        for i, hit in enumerate(response['hits']['hits']):
            uid = hit['_source'].get('uid')
            subject = data_json[uid].get('subject')
            body = data_json[uid].get('body')
            print(f"--- Result {i+1} ---")
            print(f"  Score: {hit['_score']:.4f}")
            print(f"  UID: {uid}")
            print(f"  Subject: {subject}")
            print(f"  From: {hit['_source'].get('from')}")
            print(f"  Body (partial): {body[:200]}...")  # Print only part of body
            print("-" * 20)
    else:
        print("No results found.")

# --- Main Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Search using OpenSearch and Sentence Transformers.")
    parser.add_argument("query", type=str, help="The search query text.")
    parser.add_argument(
        "--field", type=str, choices=["subject_embedding", "body_embedding"], default="subject_embedding",
        help="The embedding field to search against (default: subject_embedding)."
    )
    parser.add_argument("--top_k", type=int, default=3, help="Number of top results to retrieve (default: 3).")

    args = parser.parse_args()

    query_text = args.query
    target_field = args.field
    top_k = args.top_k

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
    query_embedding = generate_query_embedding(query_text)
    if query_embedding is None:
        print("Could not generate embedding for query. Exiting.")
        exit()

    results = perform_knn_search(query_embedding, target_field, k=top_k)
    print_search_results(results)
