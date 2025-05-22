Enron Dataset Semantic Search with OpenSearch ML Commons
========================================================

This project demonstrates how to build a semantic search solution for the Enron email dataset using OpenSearch and its Machine Learning (ML) Commons plugin for generating vector embeddings.

If you want to view the search query results, refer to the [results.md](./results.md) file.

Link to Enron Dataset: [Kaggle Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset/ "null")

1\. Data Preparation
--------------------

The initial Enron dataset typically requires cleaning and formatting for ingestion into OpenSearch.

- Data Cleaning:

  Refer to data_cleaning.ipynb or data_cleaning.py for the steps involved in cleaning the raw email data. The output of this step is a CSV file with the following columns, in this specific order:

  - `date`
  - `subject`
  - `from`
  - `to`
  - `body`
- CSV to JSON Conversion:

  The make_batches script is used to convert the cleaned CSV file into a JSON format suitable for OpenSearch ingestion. This script can produce either a single merged JSON file or a folder containing multiple JSON files, where each line represents a single email as a JSON object.
  I have already provided batches which can be ./json_batches so no need to run this.

  To run `make_batches`:

  ```
  python make_batches.py <path_to_cleaned_csv_file> <output_directory_or_file> --batch_size <optional_batch_size>

  ```

  Example JSON format (one object per line):

  ```
  {"uid": "unique_id_1", "subject": "Subject text 1", "from": "sender1@example.com", "to": "recipient1@example.com", "body": "Body text 1..."}
  {"uid": "unique_id_2", "subject": "Subject text 2", "from": "sender2@example.com", "to": "recipient2@example.com", "body": "Body text 2..."}

  ```

2\. OpenSearch Setup
--------------------

This section guides you through setting up your OpenSearch instance for ML Commons functionality.

### 2.1. OpenSearch Installation

For running OpenSearch locally, you can refer to the official documentation:

- Docker Compose: [OpenSearch Downloads - Docker Compose](https://opensearch.org/downloads.html#docker-compose "null")
- Tarball Installation: [OpenSearch Docs - Install OpenSearch (Tar)](https://opensearch.org/docs/2.5/install-and-configure/install-opensearch/tar/ "null")
- Debian Installation: [OpenSearch Docs - Install OpenSearch (Debian)](https://docs.opensearch.org/docs/latest/install-and-configure/install-opensearch/debian/)

It is recommended to use OpenSearch version 3.0 and install using the Debian package if you are on a Linux system.

### 2.2. Initial OpenSearch Configuration

After installing OpenSearch, some initial configurations are necessary for ML Commons to function optimally.

#### 2.2.1. Disable ML Node Requirement

By default, OpenSearch ML Commons expects dedicated "ML nodes" for model serving. For single-node setups or development environments, you need to disable this setting to allow models to run on your data node.

Command:

```
curl -X PUT "https://localhost:9200/_cluster/settings"\
     -u admin:yourStrongPassword@123 -k\
     -H "Content-Type: application/json"\
     -d '{
       "persistent": {
         "plugins": {
           "ml_commons": {
             "only_run_on_ml_node": false
           }
         }
       }
     }'

```

Expected Output:

```
{"acknowledged":true}

```

#### 2.2.2. Increase JVM Heap Size (Crucial for Performance)

OpenSearch is a Java application, and its performance heavily relies on the Java Virtual Machine (JVM) heap memory. Ingesting data with ML Commons (especially large text bodies and generating embeddings) is memory-intensive. Insufficient JVM heap is a common cause for timeouts and indexing errors.

Action:

1. Locate `jvm.options`:

   - For RPM/DEB installations: `/etc/opensearch/jvm.options`
   - For Tarball installations: `<OpenSearch_HOME>/config/jvm.options`
2. Edit the file: Open `jvm.options` with `sudo` (if in `/etc/`) and modify the `-Xms` (initial heap size) and `-Xmx` (maximum heap size) values.

   - Set `-Xms` and `-Xmx` to the same value. This prevents the JVM from constantly resizing the heap, which can cause performance pauses.
   - Allocate approximately 50% of your total available physical RAM to OpenSearch. This leaves sufficient memory for the operating system and file system caches.
   - Do NOT exceed 31 GB (or 32 GB) for the heap. Beyond this, the JVM's pointer compression may become less efficient. If your machine has more than 64GB RAM, cap the heap at ~31-32GB and let the remaining RAM be used by the OS for file system caching.

   Example (for a machine with 8GB RAM):

   ```
   -Xms4g
   -Xmx4g

   ```
3. Save and Close the file.
4. Restart OpenSearch:

   ```
   sudo systemctl restart opensearch

   ```
5. Verify: After restarting, you can confirm the new heap size:

   ```
   curl -X GET "https://localhost:9200/_nodes/stats/jvm?pretty"\
        -u admin:yourStrongPassword@123 -k | grep -E "heap_max_in_bytes|heap_init_in_bytes"

   ```

   Confirm that `heap_max_in_bytes` reflects your new setting.

3\. Deploy Text Embedding Model to ML Commons
---------------------------------------------

We will deploy the `huggingface/sentence-transformers/all-MiniLM-L6-v2` model, which produces 384-dimensional embeddings.

### 3.1. Register and Deploy Model

This command registers the model with ML Commons and initiates its deployment.

Command:

```
curl -X POST "https://localhost:9200/_plugins/_ml/models/_register?deploy=true"\
     -u admin:yourStrongPassword@123 -k\
     -H "Content-Type: application/json"\
     -d '{
       "name": "huggingface/sentence-transformers/all-MiniLM-L6-v2",
       "version": "1.0.1",
       "model_format": "TORCH_SCRIPT",
       "function_name": "TEXT_EMBEDDING"
     }'

```

Expected Output (Example - `task_id` will be unique):

```
{
  "task_id" : "xNAl95YBbkKtBziUlwoB",
  "status" : "CREATED"
}

```

Action: Copy the `task_id` from the response.

### 3.2. Monitor Model Deployment

Model deployment is an asynchronous process. You need to poll its status until it's `COMPLETED`.

Command (Replace `your_task_id_here` with the actual `task_id` from Step 3.1):

```
curl -X GET "https://localhost:9200/_plugins/_ml/tasks/your_task_id_here"\
     -u admin:yourStrongPassword@123 -k

```

Expected Output (Keep running until `state` is `COMPLETED`):

```
{
  "task_id": "your_task_id_here",
  "task_type": "REGISTER_MODEL",
  "function_name": "TEXT_EMBEDDING",
  "state": "COMPLETED",
  "worker_node": [
    "your_node_id_here"
  ],
  "create_time": 1747894559558,
  "last_update_time": 1747894593891,
  "is_async": true,
  "model_id": "your_new_model_id_here",  <-- **This is the new model_id you need!**
  "status": "COMPLETED",
  "output": {
    "model_id": "your_new_model_id_here"
  }
}

```

Action: Copy the `model_id` from the `COMPLETED` response. This is the ID of your newly deployed model.

4\. Create OpenSearch Ingest Pipeline
-------------------------------------

This pipeline will automatically call your deployed ML Commons model to generate embeddings for the `subject` and `body` fields every time a document is indexed into an index that uses this pipeline.

Command (Replace `your_model_id_here` with the `model_id` obtained in Step 3.2):

```
curl -X PUT "https://localhost:9200/_ingest/pipeline/text-embedding-pipeline"\
     -u admin:yourStrongPassword@123 -k\
     -H "Content-Type: application/json"\
     -d '{
       "description": "An NLP ingest pipeline for text embedding for subject and body",
       "processors": [
         {
           "text_embedding": {
             "model_id": "your_model_id_here",   <--- YOUR DEPLOYED MODEL ID
             "field_map": {
               "subject": "subject_embedding",
               "body": "body_embedding"
             }
           }
         }
       ]
     }'

```

Expected Output:

```
{"acknowledged":true}

```

5\. Create OpenSearch Index with Mappings
-----------------------------------------

This command creates the `my-email-data` index, defines its mappings (including the `knn_vector` fields for embeddings), and attaches the `text-embedding-pipeline` as its default ingest pipeline.

Important: The `dimension` for `knn_vector` must match the output dimension of your deployed model (`all-MiniLM-L6-v2` produces 384 dimensions).

Command:

```
curl -X PUT "https://localhost:9200/my-email-data"\
     -u admin:yourStrongPassword@123 -k\
     -H "Content-Type: application/json"\
     -d '{
       "settings": {
         "index.knn": true,
         "default_pipeline": "text-embedding-pipeline"
       },
       "mappings": {
         "properties": {
           "uid": { "type": "keyword" },
           "from": { "type": "keyword" },
           "to": { "type": "keyword" },
           "subject_embedding": {
             "type": "knn_vector",
             "dimension": 384,
             "space_type": "l2"
           },
           "body_embedding": {
             "type": "knn_vector",
             "dimension": 384,
             "space_type": "l2"
           }
         }
       }
     }'

```

Expected Output:

```
{"acknowledged":true,"shards_acknowledged":true,"index":"my-email-data"}

```

If the index already exists with incorrect mappings (e.g., wrong dimension), delete it first:

```
curl -X DELETE "https://localhost:9200/my-email-data?pretty"\
     -u admin:yourStrongPassword@123 -k

```

6\. Ingest Data with Python Script
----------------------------------

This Python script reads your JSON data file (one JSON object per line) and sends it to OpenSearch. The ingest pipeline will then automatically generate embeddings for the `subject` and `body` fields.

### 6.1. Python Script (`ingest.py`)

Create a file named `ingest.py` and populate it with the Python ingestion script code. This script includes logic for cleaning text and handling bulk requests.

### 6.2. How to Run the Ingestion Script

Command:

```
python ingest.py /path/to/your/json_data_file.json

```

Example:

```
python ingest.py json_batches/output_14.json

```

Expected Output during run:

```
Note: This script relies on OpenSearch ML Commons and an ingest pipeline to generate embeddings.
Reading data from 'json_output/output_14.json'...
Preparing 10000 documents for ingestion (embeddings handled by OpenSearch)...
Ingesting batches: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 100/100 [00:XX<00:00, XX.XXit/s]

Data ingestion with embeddings via ML Commons pipeline complete!
Check your OpenSearch index: my-email-data

```

(The `tqdm` progress bar will show progress, and `00:XX<00:00, XX.XXit/s` will vary based on performance).

7\. Verification and Testing
----------------------------

After running the ingestion script, use these commands to verify that documents are indexed and embeddings are created as expected.

### 7.1. Check Document Counts

Verify the total number of documents and the number of documents that successfully had `subject_embedding` and `body_embedding` fields added.

Count all documents in the index:

```
curl -X GET "https://localhost:9200/my-email-data/_count?pretty"\
     -u admin:yourStrongPassword@123 -k

```

Count documents with `subject_embedding`:

```
curl -X GET "https://localhost:9200/my-email-data/_count?pretty"\
     -u admin:yourStrongPassword@123 -k\
     -H "Content-Type: application/json"\
     -d '{
       "query": {
         "exists": {
           "field": "subject_embedding"
         }
       }
     }'

```

Count documents with `body_embedding`:

```
curl -X GET "https://localhost:9200/my-email-data/_count?pretty"\
     -u admin:yourStrongPassword@123 -k\
     -H "Content-Type: application/json"\
     -d '{
       "query": {
         "exists": {
           "field": "body_embedding"
         }
       }
     }'

```

Expected Result: All three counts should ideally match the number of documents you successfully ingested.

### 7.2. Retrieve a Sample Document

Fetch a document by its `uid` to inspect its content and confirm the presence of the `subject_embedding` and `body_embedding` fields with vector data.

Command (Replace `your_document_uid_here` with an actual `uid` from your JSON data):

```
curl -X GET "https://localhost:9200/my-email-data/_doc/your_document_uid_here?pretty"\
     -u admin:yourStrongPassword@123 -k

```

Expected Output: The response should include the original fields (`uid`, `subject`, `from`, `to`, `body`) and the newly added `subject_embedding` and `body_embedding` fields, containing arrays of floating-point numbers (your embeddings).

### 7.3. Perform a Semantic Search

This demonstrates how to perform a k-Nearest Neighbors (k-NN) search using the embeddings.

**Python Script (`semantic_search.py`):**

Create a file named `semantic_search.py` and populate it with the Python code for performing semantic searches. This script uses a local `SentenceTransformer` model to embed your query, which is then sent to OpenSearch for k-NN search against the indexed embeddings.

**How to Run Semantic Search:**

```
python semantic_search.py --query "your search query text" --field "subject_embedding" --k 3

```

**Example:**

```
python semantic_search.py "from ClickAtHome@enron.com to john.griffith@enron.com holiday  luxury shopping experience on the planet"

```

If you omit the `--query` argument, the script will prompt you to enter the search query interactively. It will also prompt for the field to search if `--field` is not provided.
