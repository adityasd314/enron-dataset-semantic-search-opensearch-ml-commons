import argparse
import json
import re
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
OPENSEARCH_PASSWORD = os.getenv('OPENSEARCH_PASSWORD', '')  # set Environment variable or set second argument as password
INDEX_NAME = os.getenv('INDEX_NAME', 'my-email-data')  # The OpenSearch index name you created
EMBEDDING_MODEL_NAME = os.getenv('EMBEDDING_MODEL_NAME', 'all-MiniLM-L6-v2')  # Added missing variable
EMBEDDINGS_DIMENSION = 384
combined_output_file = "enron_emails_combined.json"
output_folder = "json_batches"
model = SentenceTransformer(EMBEDDING_MODEL_NAME)  # only for converting the query into a embedding

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

def extract_email_addresses(query_text):
    """Extract email addresses from query text and determine if they are from/to filters."""
    # Email regex pattern
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, query_text, re.IGNORECASE)
    
    # Check for from/to keywords
    from_emails = []
    to_emails = []
    
    # Look for patterns like "from:email" or "from email"
    from_patterns = [
        r'from[:\s]+([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',
        r'sender[:\s]+([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})'
    ]
    
    # Look for patterns like "to:email" or "to email"
    to_patterns = [
        r'to[:\s]+([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})',
        r'recipient[:\s]+([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,})'
    ]
    
    for pattern in from_patterns:
        matches = re.findall(pattern, query_text, re.IGNORECASE)
        from_emails.extend(matches)
    
    for pattern in to_patterns:
        matches = re.findall(pattern, query_text, re.IGNORECASE)
        to_emails.extend(matches)
    
    # If no explicit from/to patterns found, treat all emails as general filters
    general_emails = []
    if not from_emails and not to_emails and emails:
        general_emails = emails
    
    return {
        'from_emails': list(set(from_emails)),
        'to_emails': list(set(to_emails)),
        'general_emails': list(set(general_emails))
    }

def clean_query_text(query_text):
    """Remove email addresses and from/to keywords from query text for semantic search."""
    # Remove email patterns with from/to keywords
    patterns_to_remove = [
        r'from[:\s]+[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
        r'to[:\s]+[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
        r'sender[:\s]+[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
        r'recipient[:\s]+[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # standalone emails
    ]
    
    cleaned_text = query_text
    for pattern in patterns_to_remove:
        cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text.strip()

def generate_query_embedding(query_text):
    """Generates an embedding for the search query using the local model."""
    if not query_text:
        return None
    embeddings = model.encode([query_text])
    return embeddings[0].tolist()

def build_email_filters(email_info):
    """Build OpenSearch filters for email addresses using substring matching."""
    filters = []
    
    # Add from email filters using wildcard/contains matching
    for email in email_info['from_emails']:
        filters.append({
            "bool": {
                "should": [
                    {"wildcard": {"from": f"*{email}*"}},
                    {"match_phrase": {"from": email}}
                ]
            }
        })
    
    # Add to email filters using wildcard/contains matching
    for email in email_info['to_emails']:
        filters.append({
            "bool": {
                "should": [
                    {"wildcard": {"to": f"*{email}*"}},
                    {"match_phrase": {"to": email}}
                ]
            }
        })
    
    # Add general email filters (check both from and to fields)
    for email in email_info['general_emails']:
        filters.append({
            "bool": {
                "should": [
                    {"wildcard": {"from": f"*{email}*"}},
                    {"wildcard": {"to": f"*{email}*"}},
                    {"match_phrase": {"from": email}},
                    {"match_phrase": {"to": email}}
                ]
            }
        })
    
    return filters

def perform_knn_search(query_embedding, target_field=None, k=5, email_filters=None):
    """
    Performs a k-NN search in OpenSearch with optional email filtering.
    
    Args:
        query_embedding (list): The embedding vector of the search query.
        target_field (str): The name of the knn_vector field to search against. 
                           If None, searches both subject_embedding and body_embedding.
        k (int): The number of nearest neighbors to retrieve.
        email_filters (list): List of email filter conditions.
    """
    
    if target_field:
        # Search specific field
        knn_query = {
            "knn": {
                target_field: {
                    "vector": query_embedding,
                    "k": k
                }
            }
        }
        print(f"\nSearching for query in '{target_field}'...")
    else:
        # Search both fields using bool should query
        knn_query = {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "subject_embedding": {
                                "vector": query_embedding,
                                "k": k
                            }
                        }
                    },
                    {
                        "knn": {
                            "body_embedding": {
                                "vector": query_embedding,
                                "k": k
                            }
                        }
                    }
                ]
            }
        }
        print(f"\nSearching for query in both 'subject_embedding' and 'body_embedding'...")
    
    # Build the complete query with optional email filters
    if email_filters:
        search_body = {
            "size": k,
            "query": {
                "bool": {
                    "must": [knn_query],
                    "filter": {
                        "bool": {
                            "should": email_filters
                        }
                    }
                }
            }
        }
        print(f"Applying email filters: {len(email_filters)} filter(s)")
    else:
        search_body = {
            "size": k,
            "query": knn_query
        }
    
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
    if(not os.path.exists(combined_output_file)):
    	os.system(f"cat {output_folder}/*.json > {combined_output_file}")
    data_json = load_json_data(combined_output_file)

    if response and response['hits']['hits']:
        print(f"Found {response['hits']['total']['value']} hits:\n")
        for i, hit in enumerate(response['hits']['hits']):
            uid = hit['_source'].get('uid')
            subject = data_json[uid].get('subject') if uid in data_json else hit['_source'].get('subject', 'N/A')
            body = data_json[uid].get('body') if uid in data_json else hit['_source'].get('body', 'N/A')
            print(f"--- Result {i+1} ---")
            print(f"  Score: {hit['_score']:.4f}")
            print(f"  UID: {uid}")
            print(f"  Subject: {subject}")
            print(f"  From: {hit['_source'].get('from')}")
            print(f"  To: {hit['_source'].get('to')}")
            print(f"  Body (partial): {str(body)[:200]}...")  # Print only part of body
            print("-" * 50)
    else:
        print("No results found.")

# --- Main Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Search using OpenSearch and Sentence Transformers.")
    parser.add_argument("query", type=str, help="The search query text.")
    parser.add_argument(
        "--field", type=str, choices=["subject_embedding", "body_embedding", "both"], default="both",
        help="The embedding field to search against (default: both)."
    )
    parser.add_argument("--top_k", type=int, default=3, help="Number of top results to retrieve (default: 3).")

    args = parser.parse_args()

    query_text = args.query
    target_field = args.field if args.field != "both" else None
    top_k = args.top_k

    print(f"Original query: {query_text}")
    
    # Extract email information from query
    email_info = extract_email_addresses(query_text)
    print(f"Extracted email info: {email_info}")
    
    # Clean query text for semantic search
    cleaned_query = clean_query_text(query_text)
    print(f"Cleaned query for semantic search: '{cleaned_query}'")
    
    # Generate embedding for cleaned query
    if cleaned_query:
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}...")
        query_embedding = generate_query_embedding(cleaned_query)
        if query_embedding is None:
            print("Could not generate embedding for query. Exiting.")
            exit()
    else:
        print("No semantic content found in query after cleaning.")
        query_embedding = None
    
    # Build email filters
    email_filters = build_email_filters(email_info)
    
    # If no semantic content but have email filters, perform filtered search without embeddings
    if not cleaned_query and email_filters:
        print("Performing email-only filtered search...")
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "filter": {
                        "bool": {
                            "should": email_filters
                        }
                    }
                }
            }
        }
        try:
            response = client.search(index=INDEX_NAME, body=search_body)
            print_search_results(response)
        except Exception as e:
            print(f"Error during email-filtered search: {e}")
    elif query_embedding:
        # Perform semantic search with optional email filtering
        results = perform_knn_search(query_embedding, target_field, k=top_k, email_filters=email_filters if email_filters else None)
        print_search_results(results)
    else:
        print("No valid query content found. Please provide either semantic search terms or email addresses.")
