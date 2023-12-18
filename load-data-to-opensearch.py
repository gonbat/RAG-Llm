import logging
import coloredlogs
import json
import argparse
import boto3
from utils import  secret, opensearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

coloredlogs.install(fmt='%(asctime)s %(levelname)s %(message)s', datefmt='%H:%M:%S', level='INFO')
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", type=bool, default=0)
    parser.add_argument("--index", type=str, default="rag")
    parser.add_argument("--region", type=str, default="us-east-1")
    
    return parser.parse_known_args()


def get_bedrock_client(region):
    bedrock_client = boto3.client("bedrock-runtime", region_name=region)
    return bedrock_client


def create_vector_embedding_with_bedrock(text, name, bedrock_client, links):
    payload = {"inputText": f"{text}"}
    body = json.dumps(payload)
    modelId = "amazon.titan-embed-text-v1"
    accept = "application/json"
    contentType = "application/json"

    response = bedrock_client.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )
    response_body = json.loads(response.get("body").read())

    embedding = response_body.get("embedding")
    return {"_index": name, "text": text, "vector_field": embedding, "links": links}

            
def main():
    logging.info("Starting")
    
    s3_client = boto3.client('s3')
    bucket_name = 'sagemakerdocs' 
    prefix = 'sagemaker_documentation'  # Replace with your prefix if applicable

    # List and download markdown files from S3
    objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)

    
    args, _ = parse_args()
    region = args.region
    name = args.index
    
    # Prepare OpenSearch index with vector embeddings index mapping
    logging.info("Preparing OpenSearch Index")
    opensearch_password = secret.get_secret(name, region)
    opensearch_client =  opensearch.get_opensearch_cluster_client(name, opensearch_password, region)
    
    # Check if to delete OpenSearch index with the argument passed to the script --recreate 1
    if args.recreate:
        response = opensearch.delete_opensearch_index(opensearch_client, name)
        if response:
            logging.info("OpenSearch index successfully deleted")
    
    logging.info(f"Checking if index {name} exists in OpenSearch cluster")
    exists = opensearch.check_opensearch_index(opensearch_client, name)    
    if not exists:
        logging.info("Creating OpenSearch index")
        success = opensearch.create_index(opensearch_client, name)
        if success:
            logging.info("Creating OpenSearch index mapping")
            success = opensearch.create_index_mapping(opensearch_client, name)
            logging.info(f"OpenSearch Index mapping created")
    
    # Download sagemaker docs from S3 bucket. 
    logging.info("Downloading dataset from S3")        
    all_records = []

    url_pattern = r'https://docs\.aws\.amazon\.com/[^\s,")\[\]]+'


    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    all_records = []

    for obj in objects.get('Contents', []):
        file_key = obj['Key']
        file_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = file_obj['Body'].read().decode('utf-8')

        # Extract URLs
        urls = re.findall(url_pattern, file_content)

        # Use the text_splitter to split the content
        chunks = text_splitter.split_text(file_content)
        
        # Associate each chunk with the URLs
        for chunk in chunks:
            all_records.append({'chunk': chunk, 'urls': urls.copy()})

        # Initialize bedrock client
    bedrock_client = get_bedrock_client(region)

    # Vector embedding using Amazon Bedrock Titan text embedding
    all_records_embed = []
    logging.info(f"Creating embeddings for records")

    for i, record in enumerate(all_records):
        records_with_embedding = create_vector_embedding_with_bedrock(record['chunk'], name, bedrock_client, record['urls'])
        logging.info(f"Embedding for record {i + 1} created")
        all_records_embed.append(records_with_embedding)

        if i == len(all_records) - 1:
            # Bulk put all records to OpenSearch
            success, failed = opensearch.put_bulk_in_opensearch(all_records_embed, opensearch_client)
            all_records_embed = []
            logging.info(f"Documents saved {success}, documents failed to save {failed}")

                
    logging.info("Finished creating records using Amazon Bedrock Titan text embedding")
        
    logging.info("Finished")
        
if __name__ == "__main__":
    main()