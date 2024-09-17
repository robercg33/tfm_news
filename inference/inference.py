import os
import pandas as pd
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from loader import S3DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import multiprocessing
from dotenv import load_dotenv
import subprocess
import threading
import argparse

# Load environment variables from a .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#Enable
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class InferenceProcessor:
    def __init__(
            self,
            hf_model_name,
            start_date,
            end_date,
            max_length=150,
            min_length=40,
            length_penalty=1.4,
            num_beams=4,
            halve_model=0,
            debug_mode=0,
            max_threads=4, 
            max_concurrent_files=10):
        """
        Initialize the inference processor.

        Parameters:
        - max_threads (int): Maximum number of parallel threads to use for inference.
        - max_concurrent_files (int): Maximum number of files to process concurrently.
        """
        self.loader = S3DataLoader(
            s3_bucket_name = os.getenv("S3_BUCKET_NAME"),
            dynamodb_table_name= os.getenv("DYNAMODB_TABLE_NAME"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            region_name=os.getenv("AWS_REGION"),
            start_date=start_date,
            end_date=end_date
        )
        self.max_threads = max_threads
        self.max_concurrent_files = max_concurrent_files
        self.semaphore = threading.Semaphore(max_concurrent_files)  # Initialize Semaphore with max_concurrent_files

        # Load model and tokenizer from Hugging Face Hub
        model_name = hf_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        #Check for HALVE_MODEL environment variable
        self.halve_model = (halve_model == 1)
        
        #Convert model to FP16 if HALVE_MODEL is set
        if self.halve_model:
            self.model.half()

        #Debug mode
        self.debug_mode=debug_mode

        # Set summarization parameters with default values
        self.max_length = max_length
        self.min_length = min_length
        self.length_penalty = length_penalty
        self.num_beams = num_beams


        logger.info(f"Halve model: {self.halve_model}\nDevice: {self.device}")

        self.total_records = 0  # Initialize total records

    def summarize_texts(self, texts):
        """
        Summarize a batch of texts using a pre-trained model.

        Parameters:
        - texts (list of str): The texts to summarize.

        Returns:
        - list of str: The summarized texts.
        """
        try:
            # Tokenize the inputs as a batch
            inputs = self.tokenizer(texts, return_tensors="pt", max_length=1024, truncation=True, padding=True)

            #Gett input_ids, attn_mask and max_seq len
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            # Determine the maximum sequence length in the batch
            max_len = input_ids.shape[1]

            # Check for invalid dimensions or values in input_ids and attention_mask
            for i in range(len(input_ids)):
                if input_ids[i].shape[0] != max_len or attention_mask[i].shape[0] != max_len:
                    logger.warning(f"Invalid input dimension for record {i}, replacing with default message.")
                    # Replace the invalid input with a default "failure" input
                    default_input = self.tokenizer("I cannot summarize this, record failed", return_tensors="pt", max_length=max_len, padding="max_length")
                    input_ids[i] = default_input["input_ids"].squeeze()
                    attention_mask[i] = default_input["attention_mask"].squeeze()

            # Send inputs to the device (GPU/CPU)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            #Convert inputs to FP16 if HALVE_MODEL is set
            if self.halve_model:
                inputs = {k: v.half() if v.dtype == torch.float32 else v for k, v in inputs.items()}
            
            #Syncrhonize and log memory before summarization if debug mode is active
            if self.debug_mode == 1:
                torch.cuda.synchronize()
                logger.info(f"Allocated memory before inference: {torch.cuda.memory_allocated(self.device)} bytes")
                logger.info(f"Cached memory before inference: {torch.cuda.memory_reserved(self.device)} bytes")

            # Generate the summaries
            with torch.inference_mode(): 
                summary_ids = self.model.generate(inputs["input_ids"], 
                                                max_length=self.max_length, 
                                                min_length=self.min_length, 
                                                length_penalty=self.length_penalty, 
                                                num_beams=self.num_beams, 
                                                early_stopping=True)
                # Decode the summaries
                summaries = [self.tokenizer.decode(g, skip_special_tokens=True) for g in summary_ids]

            #Synchronize and log memory after inference if debug mode is active
            if self.debug_mode == 1:
                torch.cuda.synchronize()
                logger.info(f"Allocated memory after inference: {torch.cuda.memory_allocated(self.device)} bytes")
                logger.info(f"Cached memory after inference: {torch.cuda.memory_reserved(self.device)} bytes")

            #Clear memory
            del inputs
            del summary_ids
            torch.cuda.empty_cache()

            return summaries
        except Exception as e:
            logger.error(f"Error summarizing texts: {e}")
            raise RuntimeError("Summarization failed.") # Raise an error if summarization fails

    def process_file(self, file_key, pbar, batch_size=8):
        """
        Process a single file: load it, add 'body_len' column, filter records, perform inference in batches, and save the results.

        Parameters:
        - file_key (str): The S3 key of the file to process.
        - pbar (tqdm): The progress bar object to update.
        - batch_size (int): The number of records to process in a single batch.
        """
        #Use semaphore to ensure only max_concurrent_files are executed in parallel at the same time
        with self.semaphore:
            try:
                #Load the file
                df = self.loader.download_file(file_key)
                logger.info(f"Loaded file {file_key} with {len(df)} records.")

                #Check for already summarized records
                if 'body_summary' in df.columns:
                    df['already_summarized'] = df['body_summary'].notna()
                    logger.info(f"File {file_key} has {df['already_summarized'].sum()} already summarized records.")
                else:
                    df['body_summary'] = None  # Initialize the summary column
                    df['already_summarized'] = False  # Initialize flag for summarized records

                #Add 'body_len' column and filter out records where body length exceeds 8000 characters
                df['body_len'] = df['body'].apply(len)
                df = df[df['body_len'] <= 8000]
                logger.info(f"Filtered records: {len(df)} remaining after filtering.")

                if df.empty:
                    logger.info(f"No records left to process in {file_key} after filtering.")
                    return
                
                #Sort the DataFrame by the length of the 'body' field
                df = df.sort_values(by='body_len')

                #Perform inference in batches
                for i in range(0, len(df), batch_size):
                    try:
                        #Select only unsummarized records for the current batch
                        batch_df = df.iloc[i:i + batch_size]
                        batch_texts = batch_df.loc[~batch_df['already_summarized'], 'body'].tolist()

                        #Skip empty batches
                        if not batch_texts:
                            continue

                        #Get summaries
                        batch_summaries = self.summarize_texts(batch_texts)

                        #Assign the summaries to the DataFrame
                        df.loc[batch_df.index[~batch_df['already_summarized']], 'body_summary'] = batch_summaries
                        pbar.update(len(batch_texts))  # Update the progress bar
                    except Exception as e:
                        #Log the error
                        logger.error(f"Error on index: {i}. Exception: {e}")                        

                        #IMPORTANT: Save the partially summarized file
                        self.loader.upload_file(df, file_key)
                        logger.info(f"Saved partially summarized file {file_key} with NaN for failed records.")

                        # Re-raise the exception after logging and saving
                        raise

                # Save the updated file back to S3
                self.loader.upload_file(df, file_key)
                logger.info(f"Saved summarized file {file_key}.")

                # Mark the file as processed in DynamoDB only if all records are summarized
                if df['body_summary'].isna().sum() == 0:
                    self.loader.mark_as_processed(file_key)
                    logger.info(f"Marked {file_key} as summarized in DynamoDB.")
                else:
                    logger.info(f"File {file_key} not fully summarized. Remaining records left unsummarized.")
            except RuntimeError as e:
                logger.error(f"Failed to process file {file_key}: {e}.")
                raise  # Re-raise the error to ensure the issue is not silently ignored
            except Exception as e:
                logger.error(f"Error processing file {file_key}: {e}.")
                
                raise


    def run_inference(self, batch_size=8):
        """
        Run the inference process in parallel on all files in the batch.
        """
        try:
            # Get files to process
            files_to_process = self.loader.list_unprocessed_files()

            if not files_to_process:
                logger.info("No files to process.")
                return

            # Calculate the total number of records left to summarize
            for file_key in files_to_process:
                df = self.loader.download_file(file_key)

                # Check for already summarized records
                if 'body_summary' in df.columns:
                    unsummarized_records = df['body_summary'].isna() & (df['body'].apply(len) <= 8000)
                else:
                    # If 'body_summary' does not exist, all records are unsummarized
                    unsummarized_records = df['body'].apply(len) <= 8000

                # Count the number of records left to summarize
                self.total_records += unsummarized_records.sum()

            logger.info(f"Starting inference on {len(files_to_process)} files with {self.total_records} records to summarize...")

            #Use TQDM bar to track the progress of the records
            with tqdm(total=self.total_records, desc="Overall Progress", unit="record") as pbar:
                with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
                    # Create a thread for each file to process
                    futures = [executor.submit(self.process_file, file_key, pbar, batch_size) for file_key in files_to_process]

                    # Wait for all threads to complete
                    for future in as_completed(futures):
                        try:
                            future.result()
                        except Exception as e:
                            logger.error(f"Error in parallel processing: {e}")

            logger.info("Inference process completed.")

            # Shutdown the machine after inference is completed
            logger.info("Shutting down the machine.")
            subprocess.run(["sudo", "shutdown", "-h", "now"])

        except Exception as e:
            logger.error(f"Error during inference process: {e}")
            #Shut down the machine either way
            subprocess.run(["sudo", "shutdown", "-h", "now"])

if __name__ == "__main__":
    try:
        
        ### PROGRAM ARGS ###
        # Argument parser for program arguments
        parser = argparse.ArgumentParser(description="HF summarization model inference")
        
        parser.add_argument('--hf_model_name', type=str, default='robercg33/distilbart-cnn-12-6-finetuned', help='Name of the pretrained model')
        parser.add_argument('--max_threads', type=int, default=None, help='Max concurrent threads. If none, is set to 80% of CPU machine cores')
        parser.add_argument('--start_date', type=str, default=None, help='Start date for files filter before summarize')
        parser.add_argument('--end_date', type=str, default=None, help='End date for files filter before summarize')
        parser.add_argument('--halve_model', type=int, default=0, help='Wether to halve the model (1) or not (0)')
        parser.add_argument('--debug_mode', type=int, default=0, help='Run the code in debug mode (1) or not (0)')
        parser.add_argument('--max_length', type=int, default=150, help='Maximum number of tokens for the model to generate on each summary')
        parser.add_argument('--min_length', type=int, default=40, help='Minimum number of tokens for the model to generate on each summary')
        parser.add_argument('--length_penalty', type=float, default=1.4, help='Length penalty model param')
        parser.add_argument('--num_beams', type=int, default=4, help='Num beams model param')
        parser.add_argument('--batch_size', type=int, default=10, help='Inference batch size')
        parser.add_argument('--max_concurrent_files', type=int, default=10, help="number of concurrent files to process at the same time.")

        args = parser.parse_args()

        # Set max_threads to 80% of available CPU cores
        max_threads = int(multiprocessing.cpu_count() * 0.8)

        #Create the object using the passed args
        inference_processor = InferenceProcessor(
            hf_model_name=args.hf_model_name,
            start_date=args.start_date,
            end_date=args.end_date,
            max_length=args.max_length,
            min_length=args.min_length,
            length_penalty=args.length_penalty,
            num_beams=args.num_beams,
            halve_model=args.halve_model,
            debug_mode=args.debug_mode,
            max_threads=max_threads, 
            max_concurrent_files=args.max_concurrent_files
        )

        #Perform inference
        inference_processor.run_inference(batch_size=args.batch_size)
    except Exception as e:
        logger.error(f"Failed to start inference process: {e}")
