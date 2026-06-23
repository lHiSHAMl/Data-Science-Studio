import json
from base64 import b64encode, b64decode
import zlib
import streamlit as st
import os 
import pickle
import zlib
from base64 import b64encode

def save_encoded_transformations( transformations_dict):
    """
    Save multiple transformations lists to an encoded JSON file with keys.
    
    Args:
        file_path (str): Path to save the file
        transformations_dict (dict): Dictionary of transformations lists with keys
    """
    # Convert to JSON string
    # serializable_dict = make_json_serializable(transformations_dict)
    raw_bytes = pickle.dumps(transformations_dict)
    # json_str = json.dumps(raw_bytes)
    
    # Compress and encode
    compressed = zlib.compress(raw_bytes)
    encoded = b64encode(compressed)
    # if "save_file_upload" not in st.session_state:
    downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(downloads_dir, exist_ok=True)
    fpath = os.path.join(downloads_dir, "save.enc")
    # else :  
    #     fpath = st.session_state.save_file_upload
        # Write to file
    with open(fpath, 'wb') as f:
        f.write(encoded)
    st.success(f'Your Progress has been saved to {fpath}', icon="✅")



# def save_encoded_transformations(transformations_dict):

#     # Serialize everything safely
#     raw_bytes = pickle.dumps(transformations_dict)

#     # Compress
#     compressed = zlib.compress(raw_bytes)

#     # Encode
#     encoded = b64encode(compressed)

#     st.download_button(
#         label="Download Progress",
#         data=encoded,
#         file_name="save.enc",
#         mime="application/octet-stream"
#     )

# def load_encoded_transformations(file_path):
#     """
#     Load and decode transformations lists from an encoded file.
    
#     Args:
#         file_path (str): Path to the encoded file
        
#     Returns:
#         dict: Dictionary of transformations lists with their keys
#     """
#     json_str=""
#     # Read encoded data
#     try:
#      with open(file_path, 'rb') as f:
#         encoded = f.read().decode("utf-8")
    
#     # Decode and decompress
#         compressed = b64decode(encoded)
#         json_str = zlib.decompress(compressed).decode('utf-8')
#     except FileNotFoundError as e:
#         st.write("No previous transformations found ")
#     # Convert back to Python objects
#     return json.loads(json_str) if json_str !="" else ""

def load_encoded_transformations(file_source):

    encoded_bytes = file_source.getvalue()

    compressed = b64decode(encoded_bytes)

    raw_bytes = zlib.decompress(compressed)

    pipeline_object = pickle.loads(raw_bytes)

    return pipeline_object

# def load_encoded_transformations(file_source):
#     """
#     Load and decode transformations from:
#     - file path (str)
#     - Streamlit UploadedFile
#     """

#     try:
#         # Case 1: File path
#         if isinstance(file_source, str):
#             with open(file_source, "rb") as f:
#                 encoded_bytes = f.read()

#         # Case 2: Uploaded file
#         else:
#             encoded_bytes = file_source.getvalue()  # IMPORTANT

#         # Decode base64
#         compressed = b64decode(encoded_bytes)

#         # Decompress
#         json_str = zlib.decompress(compressed).decode("utf-8")

#         return json.loads(json_str)

#     except Exception as e:
#         st.error(f"Error loading file: {e}")
#         return ""
    
import pandas as pd

def make_json_serializable(obj):
    if isinstance(obj, pd.DataFrame):
        print("Serializing DataFrame with shape:", obj)
        return {
            "__type__": "DataFrame",
            "data": obj.to_dict(orient="records"),
            "columns": list(obj.columns)
        }
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    else:
        return obj