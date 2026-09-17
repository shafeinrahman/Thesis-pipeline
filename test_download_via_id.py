import redivis
import pandas as pd
import os

def main():
    token = input("Enter token: ").strip()
    redivis.api_token = token
    try:
        df = pd.read_csv("files_list.csv")
        first_file_id = df.iloc[0]["file_id"]
        first_file_name = df.iloc[0]["file_name"]
        print(f"Testing download of {first_file_name} (ID: {first_file_id})...")
        
        file_obj = redivis.file(first_file_id)
        with file_obj.open("rb") as f:
            content = f.read()
        print("SUCCESS! File size:", len(content))
    except Exception as e:
        print("FAILED:", e)

if __name__ == "__main__":
    main()
