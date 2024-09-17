import boto3
import os
import sagemaker
from sagemaker.estimator import Estimator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define variables
role = os.getenv('SAGEMAKER_ROLE')
ecr_image = f"{os.getenv('AWS_ACCOUNT_ID')}.dkr.ecr.{os.getenv('AWS_REGION')}.amazonaws.com/{os.getenv('ECR_REPOSITORY_NAME')}:latest"
bucket_name = os.getenv('S3_BUCKET')
prefix = os.getenv('S3_PREFIX', "")
output_path = f"s3://{bucket_name}/output"
instance_type = 'ml.g4dn.4xlarge' 

model_ckpt = "robercg33/distilbart-cnn-12-6-finetuned"
model_name = "distilbart-cnn-12-6-re-finetuned"
# Define hyperparameters
hyperparameters = {
    "bucket_name": bucket_name,
    "prefix": prefix,
    "output_dir": "/opt/ml/model",
    "model_ckpt": model_ckpt,
    "model_name": model_name,
    "num_train_epochs": 10,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "test_size" : 0.1,
    "dropout" : 0.2,
    "length_penalty": 1.8,
    "max_body_length" : 5000
}


# Create a SageMaker session
sagemaker_session = sagemaker.Session()

# Define the Estimator
estimator = Estimator(
    image_uri=ecr_image,
    role=role,
    instance_count=1,
    instance_type=instance_type,
    output_path=output_path,
    sagemaker_session=sagemaker_session,
    hyperparameters=hyperparameters,
    environment={
        'S3_BUCKET': bucket_name,
        'S3_PREFIX': prefix,
    }
)

# Start the training job
estimator.fit()
