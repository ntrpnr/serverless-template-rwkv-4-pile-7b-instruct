# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from rwkvstic.load import RWKV

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model = RWKV("https://huggingface.co/Hazzzardous/RWKV-8Bit/resolve/main/RWKV-4-Pile-7B-Instruct.pqth") 

if __name__ == "__main__":
    download_model()