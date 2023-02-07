from rwkvstic.load import RWKV
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    
    model = RWKV("https://huggingface.co/Hazzzardous/RWKV-8Bit/resolve/main/RWKV-4-Pile-7B-Instruct.pqth")    

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    max_tokens = model_inputs.get('max_tokens', 200)

    model.loadContext("\n", f"Prompt: {prompt}?\n\nExpert Long Answer:")
    result = model.forward(number=max_tokens)["output"]

    # Return the results as a dictionary
    return result
