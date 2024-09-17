import boto3
import os
import pandas as pd
from io import BytesIO
from datetime import datetime

class S3DataLoader:
    """
    A class to handle loading and saving data from/to an S3 bucket and tracking the processing status with DynamoDB.
    """

    def __init__(self, s3_bucket_name, dynamodb_table_name, aws_access_key_id, aws_secret_access_key, region_name, start_date, end_date):
        """
        Initializes the S3DataLoader with the specified S3 bucket and DynamoDB table names.

        Args:
            s3_bucket_name (str): The name of the S3 bucket.
            dynamodb_table_name (str): The name of the DynamoDB table used for tracking.
        """
        #Get S3 and DynamoDB clients
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )
        self.dynamodb = boto3.resource(
            'dynamodb',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region_name
        )

        #Set attributes
        self.s3_bucket_name = s3_bucket_name
        self.dynamodb_table = self.dynamodb.Table(dynamodb_table_name)

        self.start_date = start_date
        self.end_date = end_date

        #Initialize the DynamoDB table
        self.initialize_dynamodb_entries()

    def initialize_dynamodb_entries(self):
        """
        Checks if each file in the S3 bucket is present in the DynamoDB table and adds it if missing.
        Adds start and end dates to the DynamoDB entries.
        """
        #List all the files on the bucket
        response = self.s3.list_objects_v2(Bucket=self.s3_bucket_name)
        if 'Contents' not in response:
            print("No files found in the S3 bucket.")
            return

        #For each file
        for obj in response['Contents']:
            file_key = obj['Key']

            # Extract start and end dates from the file name correctly
            date_part = file_key.split("news_", 1)[-1]  # Split by "news_" and take the last part
            date_part = date_part.rsplit('.', 1)[0]  # Remove the file extension
            file_start_date_str, file_end_date_str = date_part.split('_to_')  # Split the date part by "_to_"

            # Convert dates to "YYYY-mm-dd HH:MM:SS" format
            file_start_date = datetime.strptime(file_start_date_str, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
            file_end_date = datetime.strptime(file_end_date_str, "%Y%m%d%H%M%S").strftime("%Y-%m-%d %H:%M:%S")

            # Check if the file is already in DynamoDB
            existing_entry = self.dynamodb_table.get_item(Key={'FileKey': file_key})
            if 'Item' not in existing_entry:
                #Add the file with the start and end dates to DynamoDB
                self.dynamodb_table.put_item(
                    Item={
                        'FileKey': file_key,
                        'processed': False,
                        'date_summarized': None,
                        'StartDate': file_start_date,
                        'EndDate': file_end_date
                    }
                )
                print(f"Added {file_key} to DynamoDB table.")

    def list_unprocessed_files(self):
        """
        Lists unprocessed files in the S3 bucket by checking against the DynamoDB table.
        Filters files based on provided start and end dates from environment variables.

        Args:
            batch_size (int): The number of files to retrieve.

        Returns:
            list: A list of unprocessed file keys.
        """

        #Prepare the filter expression and attribute values
        filter_expression = '(attribute_not_exists(#proc) OR #proc = :val)'
        expression_attribute_values = {':val': False}
        expression_attribute_names = {'#proc': 'processed'}

        if self.start_date:
            filter_expression += ' AND EndDate >= :start_date'
            expression_attribute_values[':start_date'] = self.start_date

        if self.end_date:
            filter_expression += ' AND StartDate <= :end_date'
            expression_attribute_values[':end_date'] = self.end_date

        #Query unprocessed files and apply date filters
        response = self.dynamodb_table.scan(
            FilterExpression=filter_expression,
            ExpressionAttributeValues=expression_attribute_values,
            ExpressionAttributeNames=expression_attribute_names
        )

        #Extract and return the filtered file keys
        items = response.get('Items', [])
        return [item['FileKey'] for item in items]

    def download_file(self, file_key):
        """
        Downloads a file from S3 and loads it into a Pandas DataFrame.

        Args:
            file_key (str): The key of the file to download from S3.

        Returns:
            pd.DataFrame: The loaded data as a Pandas DataFrame.
        """
        #Get the file, read the parquet and return in a pandas DF
        obj = self.s3.get_object(Bucket=self.s3_bucket_name, Key=file_key)
        data = pd.read_parquet(BytesIO(obj['Body'].read()))
        return data

    def upload_file(self, data, file_key):
        """
        Uploads a DataFrame to S3 as a Parquet file.

        Args:
            data (pd.DataFrame): The DataFrame to upload.
            file_key (str): The key under which to save the file in S3.
        """
        #Uploads the file again with the summarized column added
        buffer = BytesIO()
        data.to_parquet(buffer, index=False)
        buffer.seek(0)
        self.s3.upload_fileobj(buffer, self.s3_bucket_name, file_key)

    def mark_as_processed(self, file_key):
        """
        Marks a file as processed in the DynamoDB table.

        Args:
            file_key (str): The key of the file to mark as processed.
        """
        #Query the dynamoDB table to mark the file as inferenced
        self.dynamodb_table.update_item(
            Key={'FileKey': file_key},
            UpdateExpression="SET #proc = :val, date_summarized = :date",
            ExpressionAttributeValues={
                ':val': True,
                ':date': str(pd.Timestamp.now())
            },
            ExpressionAttributeNames={'#proc': 'processed'}
        )
