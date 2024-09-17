import pandas as pd
from cleaner import Cleaner
from loader import Loader
import boto3
import os
from io import BytesIO
from dotenv import load_dotenv
import sys

#Load environment variables from .env file
load_dotenv()

#Initialize Cleaner and Loader
cleaner = Cleaner(max_length=10000, min_length=500)
loader = Loader(
    bucket_name=os.getenv('S3_COLLECTOR_BUCKET_NAME'),
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    aws_region=os.getenv('AWS_REGION')
)

def save_to_parquet(df, bucket_name, file_name, aws_access_key_id, aws_secret_access_key, aws_region):
    """
    Saves the given DataFrame to an S3 bucket in parquet format.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to save.
    bucket_name : str
        The name of the S3 bucket.
    file_name : str
        The name of the file to save.
    aws_access_key_id : str
        The AWS access key ID.
    aws_secret_access_key : str
        The AWS secret access key.
    aws_region : str
        The AWS region.

    Raises
    ------
    Exception
        If an error occurs while saving the DataFrame to S3.
    """
    try:
        s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        parquet_buffer = BytesIO()
        df.to_parquet(parquet_buffer, index=False)
        s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=parquet_buffer.getvalue())
    except Exception as e:
        print(f"An error occurred while saving the DataFrame to S3: {e}")

def get_remaining_files_count(bucket_name, s3_client):
    """
    Get the count of remaining CSV files in the bucket.

    Parameters
    ----------
    bucket_name : str
        The name of the S3 bucket.
    s3_client : boto3.client
        The S3 client used to interact with the bucket.

    Returns
    -------
    int
        The number of remaining CSV files in the bucket.
    """
    remaining_files = s3_client.list_objects_v2(Bucket=bucket_name).get('Contents', [])
    return len([item for item in remaining_files if item['Key'].endswith('.csv')])


def process_files(n_files, max_date_to_process):
    """
    Process the specified number of files from the S3 bucket.

    Parameters
    ----------
    n_files : int
        The number of files to process.
    """
    try:
        #Load CSVs from source bucket
        dataframes, file_keys = loader.load_csvs(n_files)

        if not dataframes:
            print("No dataframes loaded. Exiting.")
            return
        
        #Clean data
        for df in dataframes:
            df['body'] = df['body'].apply(cleaner.clean_text)
            df.dropna(subset=['body'], inplace=True)

        #Combine all DataFrames
        combined_df = pd.concat(dataframes, ignore_index=True)

        if combined_df.empty:
            print("Combined dataframe is empty after cleaning. Exiting.")
            return

        #Check what is the max date to process
        if isinstance(max_date_to_process, str) and max_date_to_process.lower() != "max":
            ##IF we have processed records older than the max_date_to_process
            if combined_df['date'].max() > max_date_to_process:
                #Print execution terminated and finish the execution
                print(f"Bucket cleaned up to date {max_date_to_process.strftime('%Y-%m-%d %H:%M:%S')}")
                return True

        #Create filename for parquet file
        start_date = combined_df['date'].min().strftime('%Y%m%d%H%M%S')
        end_date = combined_df['date'].max().strftime('%Y%m%d%H%M%S')
        parquet_file_name = f"news_{start_date}_to_{end_date}.parquet"

        #Save combined DataFrame to destination bucket in parquet format
        save_to_parquet(
            combined_df,
            bucket_name=os.getenv('S3_DESTINATION_BUCKET_NAME'),
            file_name=parquet_file_name,
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            aws_region=os.getenv('AWS_REGION')
        )

        #Delete processed CSVs from source bucket
        loader.delete_csvs(file_keys)

        #Inform user
        print(f"File {parquet_file_name} saved into {os.getenv('S3_DESTINATION_BUCKET_NAME')} bucket.")
        return False

    except Exception as e:
        print(f"An error occurred during processing: {e}")


def main(n_files, execution_mode, max_date_to_process):
    """
    Main function to load, clean, and save CSV files from S3.

    Parameters
    ----------
    n_files : int
        The number of files to process.
    execution_mode : str
        The mode of execution ('continuous' or 'batch').

    Raises
    ------
    Exception
        If an error occurs during processing.
    """
    try:
        #Continuous execution mode loops until the bucket is left with no CSVs files to process
        if execution_mode == "continuous":
            remaining_count = get_remaining_files_count(loader.bucket_name, loader.s3_client)
            
            while remaining_count > 0:

                stop_processing = process_files(n_files, max_date_to_process)
                if stop_processing:
                    break

                # Show remaining elements in the bucket
                remaining_count = get_remaining_files_count(loader.bucket_name, loader.s3_client)
                print(f"{remaining_count} elements remaining in the bucket.")

                if remaining_count == 0:
                    print("No more files to process. Exiting.")
                    break
        
        #Batch execution mode just do one iteration
        elif execution_mode == "batch":
            process_files(n_files)
            #Display completion of current batch
            print("Batch completed.")
        else:
            print("Invalid execution mode. Use 'continuous' or 'batch'.")
    except Exception as e:
        print(f"An error occurred during processing: {e}")


#Program execution
if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage: python executor.py <number_of_files_to_process> <execution_mode> <max_date_to_process>")
        sys.exit(1)

    #Get number of files from the argument
    try:
        n_files = int(sys.argv[1])
    except ValueError:
        print("The number_of_files_to_process parameter must be an integer.")
        sys.exit(1)

    #Get the execution mode
    execution_mode = sys.argv[2]
    if execution_mode not in ["continuous", "batch"]:
        print("execution_mode should be 'continous' or 'batch'")
        sys.exit(1)

    #Get max date to process
    try:
        max_date_to_process = sys.argv[3]
        if max_date_to_process.lower() != "max":
            max_date_to_process = pd.to_datetime(max_date_to_process, format='%Y-%m-%d %H:%M:%S')
    except Exception:
        print("<max_date_to_process must be in the format YYYY-mm-dd HH:MM:SS or 'max' to indicate processing the whole bucket")

    #Call the main function
    main(n_files, execution_mode, max_date_to_process)