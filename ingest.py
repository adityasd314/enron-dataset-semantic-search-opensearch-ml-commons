import json
import argparse
from opensearchpy import OpenSearch, RequestsHttpConnection
from tqdm import tqdm # For a progress bar
import re
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


# --- OpenSearch Client Setup ---
client = OpenSearch(
    hosts=[{'host': OPENSEARCH_HOST, 'port': OPENSEARCH_PORT}],
    http_compress=True, # Keep this enabled for performance after debugging
    http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
    use_ssl=True,
    verify_certs=False, # Set to True if using valid CA-signed certs; False for self-signed
    ssl_assert_hostname=False,
    ssl_show_warn=False,
    connection_class=RequestsHttpConnection,
    timeout=60 
)

# --- Functions ---

def read_json_data(filepath):
    """Reads JSON data from a file where each line is a JSON object."""
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from file '{filepath}': {e}")
        print("Please ensure your JSON file contains one valid JSON object per line.")
    except FileNotFoundError:
        print(f"Error: Input file '{filepath}' not found.")
    return data
def prepare_bulk_body(docs):
    """Prepares the bulk API body for OpenSearch. ML Commons Ingest Pipeline adds embeddings."""
    bulk_body = []
    for doc in docs:
        # We send the original document as is.
        # The ingest pipeline on OpenSearch will intercept this and add the embeddings.
        bulk_body.append({'index': {'_index': INDEX_NAME, '_id': doc.get('uid')}})
        bulk_body.append(doc) # Send the original doc without client-side embedding
    return bulk_body

def create_index_if_not_exists():
    """Creates the OpenSearch index with the correct mappings if it doesn't exist."""
    # Ensure the default_pipeline is set here!
    if not client.indices.exists(index=INDEX_NAME):
        print(f"Index '{INDEX_NAME}' does not exist. Creating it...")
        index_body = {
            "settings": {
                "index.knn": True, # Enable KNN for vector search
                # Crucially, attach your ingest pipeline here so it runs automatically
                "index.default_pipeline": "text-embedding-pipeline"
            },
            "mappings": {
                "properties": {
                    "uid": {"type": "keyword"},
                    "from": {"type": "keyword"},
                    "to": {"type": "keyword"},
                    # These fields will be created by the ingest pipeline
                    "subject_embedding": {
                        "type": "knn_vector",
                        "dimension": EMBEDDINGS_DIMENSION,
                        "space_type": "l2"
                    },
                    "body_embedding": {
                        "type": "knn_vector",
                        "dimension": EMBEDDINGS_DIMENSION,
                        "space_type": "l2"
                    }
                }
            }
        }
        try:
            response = client.indices.create(index=INDEX_NAME, body=index_body)
            print(f"Index creation response: {response}")
        except Exception as e:
            print(f"Error creating index: {e}")
            print("Please ensure your OpenSearch is running and accessible.")
            exit(1)
    else:
        print(f"Index '{INDEX_NAME}' already exists.")


# --- Main Script Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ingest JSON data into OpenSearch with embeddings via ML Commons pipeline.')
    parser.add_argument('input_file', type=str,
                        help='Path to the input JSON file (one JSON object per line).')
    parser.add_argument('--batch_size', type=int, default=100,
                        help='Batch size for processing documents.')
    args = parser.parse_args()

    INPUT_JSON_FILE = args.input_file

    # No local model loading needed

    create_index_if_not_exists()

    print(f"Reading data from '{INPUT_JSON_FILE}'...")
    documents_to_process = read_json_data(INPUT_JSON_FILE)

    if not documents_to_process:
        print("No documents to process. Exiting.")
        exit()

    print(f"Preparing {len(documents_to_process)} documents for ingestion...")
    
    BATCH_SIZE = args.batch_size # Adjust batch size as needed
    for i in tqdm(range(0, len(documents_to_process), BATCH_SIZE), desc="Ingesting batches"):
        batch = documents_to_process[i:i + BATCH_SIZE]
        bulk_data = prepare_bulk_body(batch)

        if not bulk_data:
            continue

        try:
            response = client.bulk(body=bulk_data)
            if response and response.get('errors'):
                print(f"Bulk indexing errors in batch {i // BATCH_SIZE + 1}:")
                for item in response.get('items', []):
                    if 'index' in item and 'error' in item['index']:
                        print(f"  Error indexing document {item['index'].get('_id')}: {item['index']['error'].get('reason')}")
        except Exception as e:
            print(f"Error during bulk indexing: {e}")

    print("\nData ingestion with embeddings via ML Commons pipeline complete!")
    print(f"Check your OpenSearch index: {INDEX_NAME}")
