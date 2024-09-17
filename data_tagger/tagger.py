import os
import pandas as pd
import random
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from utils import get_least_tagged_files, get_random_records_from_parquet, update_dynamodb_table, get_dynamodb_item, summarize_text, save_to_s3, get_last_file_index, update_last_file_index, initialize_dynamodb_with_s3_files, update_records_with_tagged_count
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def process_chunk(chunk, max_length, min_length):
    """
    Processes a chunk of data by summarizing the 'body' text.

    Parameters:
    chunk (pd.DataFrame): The chunk of data to process.
    max_length (int): Maximum length of the summary.
    min_length (int): Minimum length of the summary.

    Returns:
    pd.DataFrame: The processed chunk with summarized text.
    """
    # Use .loc to avoid SettingWithCopyWarning
    chunk.loc[:, 'body_summary'] = chunk['body'].apply(lambda text: summarize_text(text, max_length, min_length))
    return chunk

def main(max_length, min_length, num_threads):
    """
    Main function to orchestrate the summarization process.

    Parameters:
    max_length (int): Maximum length of the summary.
    min_length (int): Minimum length of the summary.
    num_threads (int): Number of parallel threads.
    """
    # Initialize DynamoDB table with elements from the source S3 bucket
    initialize_dynamodb_with_s3_files(os.getenv('S3_SOURCE_BUCKET_NAME'))

    # Update existing records in DynamoDB to include tagged_count if missing
    update_records_with_tagged_count()

    num_tagged_records = int(os.getenv('NUM_TAGGED_RECORDS'))
    records_per_source_file = int(os.getenv('RECORDS_PER_SOURCE_FILE'))
    records_per_stored_file = int(os.getenv('RECORDS_PER_STORED_FILE'))
    
    collected_records = 0
    collected_data = []
    
    with tqdm(total=num_tagged_records, desc="Collecting records") as pbar:
        while collected_records < num_tagged_records:
            file_keys = get_least_tagged_files()
            if not file_keys:
                print("No files found in the input bucket.")
                break
            
            file_key = random.choice(file_keys)
            dynamodb_item = get_dynamodb_item(file_key)
            used_records = dynamodb_item.get('UsedRecords', [])
            
            df = get_random_records_from_parquet(file_key, os.getenv('S3_SOURCE_BUCKET_NAME'), records_per_source_file, used_records)
            
            if len(df) + collected_records > num_tagged_records:
                df = df.iloc[:num_tagged_records - collected_records]
            
            records_to_use = df.index.tolist()
            update_dynamodb_table(file_key, records_to_use)
            
            collected_data.append(df)
            collected_records += len(df)
            pbar.update(len(df))
    
    if collected_records == 0:
        print("No records to process.")
        return
    
    collected_data = pd.concat(collected_data)
    
    # Get the last used file index from DynamoDB
    last_file_index = get_last_file_index()

    total_records = len(collected_data)

    # Use tqdm to display the progress of summarization
    with ThreadPoolExecutor(max_workers=num_threads) as executor, tqdm(total=total_records, desc="Summarizing records") as pbar:
        for i in range(0, total_records, records_per_stored_file):
            chunk = collected_data.iloc[i:i + records_per_stored_file].copy()
            results = list(executor.map(lambda x: process_chunk(x, max_length, min_length), [chunk]))
            processed_chunk = pd.concat(results)
            last_file_index += 1
            file_name = f"summarized_news_{last_file_index}.parquet"
            save_to_s3(processed_chunk, os.getenv('S3_DESTINATION_BUCKET_NAME'), file_name)
            pbar.update(len(chunk))
            print(f"Uploaded {file_name} to S3.")
    
    # Update the last used file index in DynamoDB
    update_last_file_index(last_file_index)
    
    print("Summarization process completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize news articles using OpenAI API and save to S3.")
    parser.add_argument('--max_length', type=int, required=True, help='Maximum length of the summary.')
    parser.add_argument('--min_length', type=int, required=True, help='Minimum length of the summary.')
    parser.add_argument('--num_threads', type=int, required=True, help='Number of parallel threads.')

    args = parser.parse_args()

    main(args.max_length, args.min_length, args.num_threads)