import joblib
import os
import pandas as pd
def create_joblib_file(df,file_name):
    os.makedirs("./joblibs", exist_ok=True)
    df  = pd.DataFrame.from_records(df)
    joblib.dump(df,f"./joblibs/{file_name}.joblib")
    return f"{file_name}.joblib created."