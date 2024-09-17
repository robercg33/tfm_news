import os
import boto3
import pandas as pd
from io import BytesIO
from dotenv import load_dotenv
import openai
import time

#Load environment variables
load_dotenv()

#AWS clients
#S3
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

#DynamoDB
dynamodb = boto3.resource(
    'dynamodb',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_REGION')
)

#Create the client that will be used for calling the API
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def initialize_dynamodb_with_s3_files(bucket_name):
    """
    Initializes the DynamoDB table with elements from the S3 bucket.
    For each element that is not in the table, inserts it with 0 tagged records and an empty tagged list.

    Parameters:
    bucket_name (str): The name of the S3 bucket.

    Returns:
    None
    """
    # Get the DynamoDB table
    table = dynamodb.Table(os.getenv('DYNAMODB_TRACKING_TABLE'))

    # Get the list of file keys from S3
    file_keys = get_file_keys_from_s3(bucket_name)

    for file_key in file_keys:
        # Check if the file_key exists in DynamoDB
        response = table.get_item(Key={'FileKey': file_key})
        if 'Item' not in response:
            # If the file_key does not exist, insert it with initial values
            table.put_item(Item={
                'FileKey': file_key,
                'UsedRecords': [],
                'tagged_count': 0
            })
            print(f"Initialized {file_key} in DynamoDB.")

def update_records_with_tagged_count():
    """
    Updates existing DynamoDB records to include the 'tagged_count' attribute.

    This function scans the DynamoDB table for items that do not have the 'tagged_count' attribute.
    For each item found, it calculates the length of the 'UsedRecords' list and updates the item with the 'tagged_count' attribute,
    setting its value to the length of the 'UsedRecords' list.

    Returns:
    None
    """
    # Get the DynamoDB table
    table = dynamodb.Table(os.getenv('DYNAMODB_TRACKING_TABLE'))

    # Scan the table to get all items without 'tagged_count' attribute
    response = table.scan(
        FilterExpression='attribute_not_exists(tagged_count)'
    )

    items = response.get('Items', [])

    for item in items:
        file_key = item['FileKey']
        used_records = item.get('UsedRecords', [])

        # Update the item with the 'tagged_count' attribute
        table.update_item(
            Key={'FileKey': file_key},
            UpdateExpression="SET tagged_count = :count",
            ExpressionAttributeValues={
                ':count': len(used_records)
            }
        )
        print(f"Updated {file_key} with tagged_count = {len(used_records)}")

def get_file_keys_from_s3(bucket_name):
    """
    Retrieves the list of file keys from the S3 bucket.

    Parameters:
    bucket_name (str): The name of the S3 bucket.

    Returns:
    list: The list of file keys.
    """
    #Get the whole list of files in S3 bucket
    response = s3_client.list_objects_v2(Bucket=bucket_name)
    #Filter only the parquet files
    files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.parquet')]
    return files

def get_random_records_from_parquet(file_key, bucket_name, num_records, used_records):
    """
    Retrieves a random sample of records from a Parquet file stored in S3, excluding already used records.

    Parameters:
    file_key (str): The key of the Parquet file in the S3 bucket.
    bucket_name (str): The name of the S3 bucket.
    num_records (int): The number of records to retrieve.
    used_records (list): The list of indices of records that have already been used.

    Returns:
    pd.DataFrame: A DataFrame containing the sampled records.
    """
    #Get the selected parquet object into a DF
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)

    #Read the content into a BytesIO object
    parquet_file = BytesIO(response['Body'].read())

    df = pd.read_parquet(parquet_file)

    #Exclude used records
    df = df[~df.index.isin(used_records)]

    #Return a random sample of num_records. If the DF has less elements than num_records, return the whole DF
    return df.sample(n=num_records) if len(df) > num_records else df

def update_dynamodb_table(file_key, used_records):
    """
    Updates the DynamoDB table with the indices of records that have been processed and 
    updates the count of tagged records for each file.

    Parameters:
    file_key (str): The key of the file in the S3 bucket.
    used_records (list): A list of indices of records that have been processed.

    Returns:
    dict: The response from the DynamoDB update operation.
    """
    # Get the DynamoDB table
    table = dynamodb.Table(os.getenv('DYNAMODB_TRACKING_TABLE'))

    # Update the DynamoDB item corresponding to the file_key
    response = table.update_item(
        Key={'FileKey': file_key},
        UpdateExpression="""
            SET UsedRecords = list_append(if_not_exists(UsedRecords, :empty_list), :new_records),
                tagged_count = if_not_exists(tagged_count, :zero) + :new_count
        """,  # Append new records to the UsedRecords list and update the tagged_count
        ExpressionAttributeValues={
            ':new_records': used_records,  # New records to append
            ':empty_list': [],  # Empty list if UsedRecords does not exist
            ':zero': 0,  # Zero value if tagged_count does not exist
            ':new_count': len(used_records)  # Number of new records to add to tagged_count
        },
        ReturnValues="UPDATED_NEW"  # Return the newly updated attributes
    )
    return response

def get_least_tagged_files(limit=1):
    """
    Retrieves the file keys with the least number of tagged records from DynamoDB.

    Parameters:
    limit (int): The number of files to retrieve. Defaults to 1.

    Returns:
    list: The list of file keys with the least number of tagged records.
    """
    # Get the DynamoDB table
    table = dynamodb.Table(os.getenv('DYNAMODB_TRACKING_TABLE'))

    # Scan the table to get FileKey and tagged_count attributes where tagged_count exists
    response = table.scan(
        ProjectionExpression='FileKey, tagged_count',  # Only retrieve these attributes
        FilterExpression='attribute_exists(tagged_count)',  # Only include items where tagged_count exists
    )

    # Retrieve the items from the response
    items = response.get('Items', [])

    # Exclude the 'last_file_index' entry
    items = [item for item in items if item['FileKey'] != 'last_file_index']

    # Sort items based on tagged_count in ascending order
    items.sort(key=lambda x: x['tagged_count'])

    # Extract the FileKeys of the least tagged files up to the specified limit
    least_tagged_files = [item['FileKey'] for item in items[:limit]]

    return least_tagged_files

def get_dynamodb_item(file_key):
    """
    Retrieves an item from the DynamoDB table.

    Parameters:
    file_key (str): The key of the file in the S3 bucket.

    Returns:
    dict: The item retrieved from the DynamoDB table.
    """
    #Get dynamo tracking table
    table = dynamodb.Table(os.getenv('DYNAMODB_TRACKING_TABLE'))
    #Get the item corresponding to the file name and return it
    response = table.get_item(Key={'FileKey': file_key})
    return response.get('Item', {})

def get_last_file_index():
    """
    Retrieves the last used file index from the DynamoDB table.

    Returns:
    int: The last used file index. Defaults to 0 if not found.
    """
    table = dynamodb.Table(os.getenv('DYNAMODB_TRACKING_TABLE'))
    response = table.get_item(Key={'FileKey': 'last_file_index'})
    return response.get('Item', {}).get('Index', 0)

def update_last_file_index(new_index):
    """
    Updates the last used file index in the DynamoDB table.

    Parameters:
    new_index (int): The new file index to be stored.

    Returns:
    None
    """
    table = dynamodb.Table(os.getenv('DYNAMODB_TRACKING_TABLE'))
    table.put_item(Item={'FileKey': 'last_file_index', 'Index': new_index})

def summarize_text(text, max_length, min_length):
    """
    Summarizes a given text using the OpenAI API.

    Parameters:
    text (str): The text to summarize.
    max_length (int): The maximum length of the summary.
    min_length (int): The minimum length of the summary.

    Returns:
    str: The summarized text.
    """

    backoff_time = 1  # Initial backoff time in seconds
    max_backoff_time = 30  # Maximum backoff time in seconds
    retry_attempts = 2  # Number of retry attempts

    for attempt in range(retry_attempts):
        try:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You will receive news articles from webpages in raw text and make a summary of that news with the key concepts of the article. "
                                   "Do not explain the article in third person, but just summarize. "
                                   "Instead of saying `The article talks about the increase in price of X and...`, say `The price of X increases and...`"
                    },
                    {
                        "role": "system",
                        "content": f"The summary MUST have between {min_length} and {max_length} CHARACTERS."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize this article: {text}"
                    }
                ]
            )
            return completion.choices[0].message.content
        
        except Exception as e:
            print(f"Exception in the API loop: {e}")
            if attempt < retry_attempts - 1:
                time.sleep(backoff_time)
                backoff_time = min(backoff_time * 2, max_backoff_time)
            else:
                print("Stopping summarization after max retries.")
                return None

    # If the function reaches this point, it means all attempts have failed
    return None

def save_to_s3(df, bucket_name, file_name):
    """
    Saves a DataFrame as a Parquet file to an S3 bucket.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    bucket_name (str): The name of the S3 bucket.
    file_name (str): The name of the file to save in the S3 bucket.
    """
    parquet_buffer = BytesIO()
    df.to_parquet(parquet_buffer, index=False)
    s3_client.put_object(Bucket=bucket_name, Key=file_name, Body=parquet_buffer.getvalue())