import  numpy as np
import pandas as pd
from src.create_chunks import  semantic_chunk_text
import requests
import json
def create_embedding(text_list):
    r  = requests.post("http://localhost:11434/api/embed",json = {
        "model" : "bge-m3",
        "input" :text_list
    })
    return r.json()["embeddings"]

def create_embedded_df(json_chunks):

    text_list = [c['text'] for c in json_chunks['chunks'] ]

    embeddings  = create_embedding(text_list)

    for i,chunk in enumerate(json_chunks['chunks']):
        chunk["embedding"] = embeddings[i]
        
    return json_chunks