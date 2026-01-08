from src.create_chunks import semantic_chunk_text
from src.process_embedding import create_embedded_df
from src.create_joblib import create_joblib_file
import pandas as pd
import json
import os
files  = os.listdir("./data/")

for file in files:
    with open(f"data/{file}","r",encoding="utf-8") as f:
        text  = f.read()
    # json_chunks creation
    json_chunks  = semantic_chunk_text(text)
    file_rename = file.split(".t")[0]
    # json file creation
    with open(f"json_data/{file_rename}.json","w",encoding="utf-8") as f:
        json.dump(json_chunks,f)
    df = create_embedded_df(json_chunks)

    


    print(f"done:{file} to {file_rename}.json")

    # create joblib file
    joblib_message = create_joblib_file(df,file_rename)
    print(joblib_message)


print("all done")
