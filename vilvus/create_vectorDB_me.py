
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


connection = connections.connect(host="localhost", port=19530)
mc = MilvusClient(connections=connection)


model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")


def create_milvus_collection(collection_name, dim):
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    fields = [
    FieldSchema(name='id', dtype=DataType.INT64, description='ids', max_length=100, is_primary=True, auto_id=False),
    FieldSchema(name='embedding', dtype=DataType.FLOAT_VECTOR, description='embedding vectors', dim=dim),
    FieldSchema(name="document", dtype=DataType.VARCHAR, max_length=60000),
    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=20000),
    FieldSchema(name="metadata_position", dtype=DataType.INT64, description="An integer field"),
    ]
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=collection_name, schema=schema)

    # create IVF_FLAT index for collection.
    index_params = {#this is the info for how the index is created, look up the indexing method
        'metric_type':'L2',
        'index_type':"IVF_FLAT",
        'params':{"nlist":2048}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Successfully created collection: `{collection_name}`")
    return collection

collection = create_milvus_collection('judgement_db', 1024)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=100)

df=pd.read_csv('complete_data.csv')
data=[]
db_id=0

for i in tqdm(range(0, 50)):
    try:
        print("encoding document",i+1)
        file_name=df['file_name'][i]
        file_content=df['content'][i]
        text_chunks=text_splitter.split_text(file_content)
        counter=0
        for chunk in text_chunks:
            counter+=1
            embedding = model.encode(chunk)
            document = {
                "id": db_id,
                "embedding": embedding,
                "document": chunk,
                "metadata": file_name,
                "metadata_position": counter
            }
            db_id+=1
            data.append(document)
    except:
        print("error in encoding document",i+1)
        
collection.insert(data)
collection.flush()