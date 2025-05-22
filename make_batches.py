import csv
import json
from datetime import datetime
import sys
import os
import argparse

csv.field_size_limit(sys.maxsize)
DEFAULT_INPUT_FILE = "cleaned_data.csv"
DEFAULT_OUTPUT_FOLDER = "json_batches"
DEFAULT_OUTPUT_FILE = "enron_emails_combined.json"
DEFAULT_BATCH_SIZE = 10000
def convert_csv_to_json(csv_file_path, output_dir, max_entries_per_file=DEFAULT_BATCH_SIZE):
    """
    Converts a CSV file of Enron emails into multiple JSON (NDJSON) files, each with a maximum number of entries.

    Args:
        csv_file_path (str): Path to the input CSV file.
        output_dir (str): Path to the output folder where JSON files will be saved.
        max_entries_per_file (int): Maximum number of entries per JSON file.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_count = 0
    file_count = 1
    current_file = None

    FIELD_ORDER = {
        0: "date",
        1: "subject",
        2: "from",
        3: "to",
        4: "body"
    }

    print(f"Starting conversion of entries from '{csv_file_path}' into '{output_dir}/' folder...")

    try:
        with open(csv_file_path, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            # next(reader, None)  # Uncomment if header is present

            for row in reader:
                if not row or len(row) < 5:
                    print(f"Skipping malformed row: {row}", file=sys.stderr)
                    continue

                if processed_count % max_entries_per_file == 0:
                    if current_file:
                        current_file.close()
                    output_path = os.path.join(output_dir, f"output_{file_count}.json")
                    current_file = open(output_path, 'w', encoding='utf-8')
                    print(f"Writing to {output_path}...")
                    file_count += 1

                email_doc = {
                    "uid": f"{processed_count + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "subject": row[1].strip() if row[1] else "",
                    "from": row[2].strip() if row[2] else "",
                    "to": row[3].strip() if row[3] else "",
                    "body": row[4].strip() if row[4] else ""
                }

                current_file.write(json.dumps(email_doc) + '\n')
                processed_count += 1

        if current_file:
            current_file.close()

        print(f"Finished processing {processed_count} entries into {file_count - 1} files.")

    except FileNotFoundError:
        print(f"Error: CSV file not found at '{csv_file_path}'", file=sys.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a CSV file of Enron emails into JSON batches.")
    parser.add_argument("--input_csv_file", type=str, default=DEFAULT_INPUT_FILE, help="Path to the input CSV file.")
    parser.add_argument("--output_folder", type=str, default=DEFAULT_OUTPUT_FOLDER, help="Output folder for JSON files.")
    parser.add_argument("--max_entries_per_file", type=int, default=DEFAULT_BATCH_SIZE, help="Max entries per JSON file.")
    parser.add_argument("--output_file", type=str, default=DEFAULT_OUTPUT_FILE, help="Output file for combined JSON.")


    args = parser.parse_args()
    output_folder = args.output_folder
    input_csv_file = args.input_csv_file
    combined_output_file = args.output_file
    convert_csv_to_json(input_csv_file, output_folder)

    # combine all into one file (if GPU is not a problem)
    os.system(f"cat {output_folder}/*.json > {combined_output_file}")
    print(f"Combined JSON file created at {combined_output_file}")