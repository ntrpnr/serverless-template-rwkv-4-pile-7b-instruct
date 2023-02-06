from prwkv.rwkvtokenizer import RWKVTokenizer
from prwkv.rwkvrnnmodel import RWKVRNN4NeoForCausalLM
import torch

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global model
    global tokenizer
    
    device = 0 if torch.cuda.is_available() else -1

    tokenizer = RWKVTokenizer.default()
    model = RWKVRNN4NeoForCausalLM.from_pretrained("RWKV-4-430M") # options RWKV-4-1B5 RWKV-4-3B RWKV-4-7B  RWKV-4-14B
    

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    if prompt == None:
        return {'message': "No prompt provided"}

    max_length = model_inputs.get('max_length', 32)
    repetition_penalty = model_inputs.get('repetition_penalty', 0.0)
    temperature = model_inputs.get('temperature', 0.8)

    context_input_ids = tokenizer.encode(prompt).ids
    model.warmup_with_context(context=context_input_ids)

    def callback(ind):
        token = tokenizer.decode([ind],skip_special_tokens=False)
        print(token,end="")

    ctx = model.generate(input_ids=[],streaming_callback=callback,max_length=max_length,repetition_penalty=repetition_penalty,temperature=temperature,stop_on_eos=True)
    result = tokenizer.decode(ctx,skip_special_tokens=False) # cpu 3 tokens a second


    # Return the results as a dictionary
    return result
