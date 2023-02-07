# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

import requests
import shutil

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    # download https://huggingface.co/Hazzzardous/RWKV-8Bit/resolve/main/RWKV-4-Pile-7B-Instruct.pqth to models/RWKV-4-Pile-7B-Instruct.pqth using wget

    download_file("https://huggingface.co/Hazzzardous/RWKV-8Bit/resolve/main/RWKV-4-Pile-7B-Instruct.pqth")

def download_file(url):
    local_filename = url.split('/')[-1]
    with requests.get(url, stream=True) as r:
        with open(local_filename, 'wb') as f:
            shutil.copyfileobj(r.raw, f)

    return local_filename





if __name__ == "__main__":
    download_model()