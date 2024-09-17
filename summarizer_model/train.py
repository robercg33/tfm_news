import logging
import os
from datasets import Dataset, load_metric
import s3fs
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, AutoConfig
from huggingface_hub import HfApi, create_repo
from dotenv import load_dotenv
import json
import torch

logging.basicConfig(level=logging.INFO)

# Setup logging to output to stdout
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

# Load environment variables from .env file
load_dotenv()

# Function to load hyperparameters from the JSON file
def load_hyperparameters():
    with open('/opt/ml/input/config/hyperparameters.json') as f:
        hyperparameters = json.load(f)
    return hyperparameters

def load_data_from_s3(bucket_name, prefix, max_body_length=5000):
    """
    Load Parquet files from an S3 bucket and convert to a HuggingFace dataset.
    
    Parameters:
    bucket_name (str): The name of the S3 bucket.
    prefix (str): The prefix (folder path) where the Parquet files are stored.
    
    Returns:
    Dataset: A HuggingFace dataset.
    """
    logger.info("Loading data from S3 bucket...")
    s3 = s3fs.S3FileSystem()
    files = s3.glob(f's3://{bucket_name}/{prefix}*.parquet')

    # Load parquet files into pandas DataFrame
    dataframes = [pd.read_parquet(f"s3://{file}") for file in files]
    
    try:
      df = pd.concat(dataframes, ignore_index=True) #This somehow can unexpectedly fail
    except:
      df = pd.concat([d.transpose() for d in dataframes], axis=1, ignore_index=True).T

    #Create the length of the summary column
    df["len_summary"] = df["body_summary"].apply(len)

    #Filter out all summaries smaller than 200 and bigger than 800
    df = df[(df["len_summary"] >= 200) & (df["len_summary"] <= 800)]

    #Create the body length column
    df["len_body"] = df["body"].apply(len)

    #Filter out all bodies that are longer than 5000 characters
    df = df.loc[df["len_body"] <= max_body_length]

    # Convert to HuggingFace dataset
    dataset = Dataset.from_pandas(df)
    logger.info("Data loaded successfully.")
    return dataset

def transform_data(dataset, tokenizer, device):
    """
    Apply necessary transformations to the dataset.
    
    Parameters:
    dataset (Dataset): The HuggingFace dataset to transform.
    
    Returns:
    Dataset: The transformed dataset.
    """
    logger.info("Transforming dataset...")

    def preprocess(example):
        return {
            'input_ids': torch.tensor(tokenizer(example['body'], truncation=True, padding='max_length', max_length=512).input_ids).to(device),
            'labels': torch.tensor(tokenizer(example['body_summary'], truncation=True, padding='max_length', max_length=128).input_ids).to(device)
        }

    transformed_dataset = dataset.map(preprocess, batched=True)
    logger.info("Dataset transformed successfully.")
    return transformed_dataset

def cast_to_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
    
def cast_to_float(value, default=0.0):
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def compute_metrics(eval_pred, tokenizer):
    metric = load_metric("rouge")
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them
    decoded_labels = [[label if label != -100 else tokenizer.pad_token_id for label in label_seq] for label_seq in labels]

    # Compute ROUGE scores
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    return result

def main():
    
    # Load hyperparameters
    hyperparameters = load_hyperparameters()
    #logger.info(f"Hyperparameters: {hyperparameters}")

    # Assign hyperparameters
    bucket_name = hyperparameters.get('bucket_name',"")
    prefix = hyperparameters.get("prefix", "")
    model_name = hyperparameters.get("model_name","bart-large")
    output_dir = hyperparameters.get('output_dir', '/opt/ml/model')
    model_ckpt = hyperparameters.get('model_ckpt', 'facebook/bart-large')
    num_train_epochs = cast_to_int(hyperparameters.get('num_train_epochs', 3))
    train_batch_size = cast_to_int(hyperparameters.get('train_batch_size', 8))
    eval_batch_size = cast_to_int(hyperparameters.get('eval_batch_size', 8))
    learning_rate = cast_to_float(hyperparameters.get('learning_rate', 5e-5))
    weight_decay = cast_to_float(hyperparameters.get('weight_decay', 0.01)) 
    test_size = cast_to_float(hyperparameters.get("test_size", 0.1))
    dropout = cast_to_float(hyperparameters.get("dropout", 0.1))
    length_penalty = cast_to_float(hyperparameters.get("length_penalty",1.5))
    max_body_length = cast_to_int(hyperparameters.get("max_body_length", 5000))

    # Load dataset from S3
    dataset = load_data_from_s3(bucket_name, prefix, max_body_length = max_body_length)
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    #Load device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transform dataset
    dataset = transform_data(dataset, tokenizer, device)

    # Split dataset into train and validation
    train_test_split = dataset.train_test_split(test_size=test_size)
    train_dataset = train_test_split['train']
    eval_dataset = train_test_split['test']

    #Load the configuration and set parameters
    model_config = AutoConfig.from_pretrained(
        model_ckpt, 
        dropout=dropout, 
        attention_dropout=dropout, 
        length_penalty=length_penalty
    )

    # Load model into the device
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt, config=model_config).to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        eval_strategy="epoch",
        logging_dir=f'{output_dir}/logs',
        save_total_limit=3,
        logging_strategy="epoch",
        weight_decay=weight_decay
        #gradient_accumulation_steps=4  # Use gradient accumulation
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )

    # Train model
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training finished.")

    # Save model and tokenizer to the output directory
    logger.info("Saving model and tokenizer...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Model and tokenizer saved successfully.")

    # Push to Hugging Face Hub
    #Create the repo first if not exists
    repo_name = f"{os.environ['HF_USERNAME']}/{model_name}-finetuned"
    logger.info(f"Repo name {repo_name}")
    try:
        create_repo(repo_name, token=os.environ['HF_API_TOKEN'])
    except Exception as e:
        print(f"When creating the repo: {e}")

    #Upload the model to the created repo
    api = HfApi()
    api.upload_folder(
        folder_path=output_dir,
        repo_id=repo_name,
        repo_type="model",
        commit_message="Initial commit of fine-tuned model",
        token=os.environ["HF_API_TOKEN"]
    )

if __name__ == "__main__":
    main()