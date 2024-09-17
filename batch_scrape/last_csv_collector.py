import pandas as pd
import requests
import os
from datetime import datetime
import boto3
import json
import logging
from dotenv import load_dotenv

#Load environment variables from .env file
load_dotenv()

#Configure AWS clients with environment variables
s3_client = boto3.client(
    's3',
    region_name=os.getenv('AWS_REGION')
)
lambda_client = boto3.client(
    'lambda',
    region_name=os.getenv('AWS_REGION')
)

col_names = ["ID", "Hash", "urls_csv"]
url = "http://data.gdeltproject.org/gdeltv2/masterfilelist.txt"
url_col_idx = 60

#Configure the logger
logging.basicConfig(
    level=logging.INFO,  # Set the logging level
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # Define the log format
    handlers=[logging.StreamHandler()]  # Ensure logs are sent to stdout
)

logger = logging.getLogger(__name__)

def main():
    """
    Main function to fetch the latest GDELT CSV, process it, and upload the results to an S3 bucket.
    
    Environment Variables:
    - AWS_REGION: AWS region
    - S3_COLLECTOR_BUCKET_NAME: Name of the S3 bucket
    - LAMBDA_SCRAPER_FUNCTION_NAME: Name of the AWS Lambda function for scraping URLs

    Returns:
    None
    """

    # Get the list of bucket names from environment variable
    bucket_names = os.getenv('S3_COLLECTOR_BUCKET_NAMES').split(',')

    # Make the request to download the file
    response = requests.get(url)

    # Check if the request was successful
    if response.status_code == 200:
        # Get the text content

        logger.info("Request fetched successfully")

        try:
            text_content = response.text

            # Filename for temp file
            filename = f"urls_{datetime.now().strftime('%Y_%m_%d__%H_%M_%S')}.csv"
            
            # Save the text content to a local file
            with open(filename, 'w') as file:
                file.write(text_content)

            # Get the URL of the current time new published DF
            url_last_csv = pd.read_csv(filename, sep=" ", header=None, names=col_names).iloc[-3]["urls_csv"]

            # Extract date from url_last_csv
            date_str = url_last_csv.split('/')[-1].split('.')[0]
            date = datetime.strptime(date_str, "%Y%m%d%H%M%S")

            # Delete the aux csv file
            os.remove(filename)

            # Get the new URL list (without duplicates)
            url_list = pd.read_csv(url_last_csv, delimiter='\t', header=None)[url_col_idx].unique().tolist()

            logger.info("URLs list fetched, calling lambda scraper.")

            # Call created lambda function with the list of urls "url_list"
            response = lambda_client.invoke(
                FunctionName=os.getenv('LAMBDA_SCRAPER_FUNCTION_NAME'),
                InvocationType='RequestResponse',
                Payload=json.dumps({"urls": url_list})
            )  

            logger.info("Lambda call successful")

            # Get the response payload
            response_payload = json.load(response['Payload'])

            # Convert the response payload from string to a list of dictionaries
            response_list = json.loads(response_payload)
            # Create the DF to store into S3
            df_for_s3 = pd.DataFrame(response_list)

            # Drop the rows with NaN values
            df_for_s3 = df_for_s3.dropna()

            # Save the response from the lambda function into a csv in S3
            result_filename = f"news_{date.strftime('%Y_%m_%d__%H_%M_%S')}.csv"
            df_for_s3.to_csv(result_filename, index=False)

            logger.info("Uploading to S3...")

            # Upload to all specified S3 buckets
            for bucket_name in bucket_names:
                s3_client.upload_file(result_filename, bucket_name, result_filename)
                logger.info(f"Uploaded to S3 bucket: {bucket_name}")

            logger.info("Uploaded to S3")

            # Delete the local result file
            os.remove(result_filename)

        except Exception as e:
            logger.error(f"An exception occurred: {e}")
            exit(0)
    else:
        logger.error(f"Failed to retrieve the file. Status code: {response.status_code}")
        exit(0)

if __name__ == "__main__":
    main()