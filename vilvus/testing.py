import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings)
from langchain_community.vectorstores import milvus
import _csv
from tqdm import tqdm
from langchain_community.vectorstores import Milvus
import pymilvus
from pymilvus import (
    MilvusClient, utility, connections,
    FieldSchema, CollectionSchema, DataType, IndexType,
    Collection, AnnSearchRequest, RRFRanker, WeightedRanker, db
)
import time

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
query=model.encode("What is the court's standing on discrimination related to sex workers and their children")

client = MilvusClient()
client.load_collection(collection_name="judgement_db")

search_params = {
    "nprobe": 10,  # Number of segments to search in each probe
    "metric_type": "L2",  # Must match the index metric_type
    "params": {},
}
results = client.search(
    collection_name="judgement_db",
    data=[query.tolist()],
    #anns_field="embedding", 
    #param=search_params, 
    limit=10  # Number of nearest neighbors to retrieve
)

# Print the results
for result in results[0]:
    print(result['id'])

with open('related_docs.txt', 'w') as f:
    for result in results[0]:
        f.write(f"{result['id']}\n")