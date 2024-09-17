## The loader script creates a loader class that is used to connect to an S3 bucket. 

import boto3
import pandas as pd
from io import StringIO
from datetime import datetime

class Loader:
    """
    A class used to load and delete CSV files from an S3 bucket.
    
    Attributes
    ----------
    bucket_name : str
        The name of the S3 bucket.
    s3_client : boto3.client
        The S3 client used to interact with the bucket.
    
    Methods
    -------
    load_csvs(n_files)
        Loads the last n_files CSV files from the S3 bucket and adds a date column.
    delete_csvs(files_to_delete)
        Deletes the specified files from the S3 bucket.
    """
    def __init__(self, bucket_name, aws_access_key_id, aws_secret_access_key, aws_region):
        """
        Parameters
        ----------
        bucket_name : str
            The name of the S3 bucket.
        aws_access_key_id : str
            The AWS access key ID.
        aws_secret_access_key : str
            The AWS secret access key.
        aws_region : str
            The AWS region.
        """
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        self.bucket_name = bucket_name

    def load_csvs(self, n_files):
        """
        Loads the last n_files CSV files from the S3 bucket and adds a date column.
        
        Parameters
        ----------
        n_files : int
            The number of files to load.
        
        Returns
        -------
        list of pd.DataFrame
            A list of DataFrames containing the loaded data.
        list of str
            A list of the file keys for the loaded files.
        
        Raises
        ------
        ValueError
            If no files are found in the bucket.
        """
        try:
            #List files in the bucket
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            files = [item['Key'] for item in response.get('Contents', []) if item['Key'].endswith('.csv')]
            
            if not files:
                raise ValueError("No CSV files found in the bucket.")
            
            #Sort files by name to get the oldest first
            files.sort()

            #Adjust n_files if there are fewer files than n_files
            if len(files) < n_files:
                n_files = len(files)
            
            #Load the specified number of files
            dataframes = []
            loaded_files = files[:n_files]
            for file_key in loaded_files:
                csv_obj = self.s3_client.get_object(Bucket=self.bucket_name, Key=file_key)
                body = csv_obj['Body'].read().decode('utf-8')
                df = pd.read_csv(StringIO(body), lineterminator='\n')
                
                #Extract date from filename and add as a column

                #First get the name without the extension
                filename_wo_extension = file_key.split(".")[0]
                #And then, remove the "news_" part
                date_str = filename_wo_extension.replace("news_","")
                #Now convert to date, and add as a column
                file_date = datetime.strptime(date_str, '%Y_%m_%d__%H_%M_%S')
                df['date'] = file_date

                #And finally, drop NaN values and duplicated bodies
                df = df.dropna()
                df = df.drop_duplicates(subset="body")
                
                #Now, append to the list of dataframes
                dataframes.append(df)
            
            return dataframes, loaded_files
        
        except Exception as e:
            print(f"An error occurred while loading CSV files: {e}")
            #Return empty lists
            return [], []

    def delete_csvs(self, files_to_delete):
        """
        Deletes the specified files from the S3 bucket.
        
        Parameters
        ----------
        files_to_delete : list of str
            The list of file keys to delete.
        
        Raises
        ------
        Exception
            If an error occurs while deleting the files.
        """
        try:
            #Delete the specified files from the bucket
            for file_key in files_to_delete:
                self.s3_client.delete_object(Bucket=self.bucket_name, Key=file_key)
        except Exception as e:
            print(f"An error occurred while deleting CSV files: {e}")