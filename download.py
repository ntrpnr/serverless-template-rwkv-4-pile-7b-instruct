# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from prwkv.rwkvtokenizer import RWKVTokenizer
from prwkv.rwkvrnnmodel import RWKVRNN4NeoForCausalLM

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    tokenizer = RWKVTokenizer.default()
    model = RWKVRNN4NeoForCausalLM.from_pretrained("RWKV-4-7B") # options RWKV-4-1B5 RWKV-4-3B RWKV-4-7B  RWKV-4-14B

if __name__ == "__main__":
    download_model()