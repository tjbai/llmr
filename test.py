# %%
from datasets import load_dataset

# %%
dataset = load_dataset('google-research-datasets/mbpp')

# %%
import torch
from transformers import pipeline
pipe = pipeline('text-generation', 'meta-llama/Llama-3.2-1B-Instruct', torch_dtype=torch.float16)

# %%
messages = [{'role': 'user', 'content': 'hello how are you today'}]
generate_kwargs = {'num_return_sequences': 2, 'do_sample': True, 'temperature': 0.7}
resp = pipe(messages, max_new_tokens=32, **generate_kwargs)

# %%
print(resp[0]['generated_text'][-1]['content'])
print(resp[1]['generated_text'][-1]['content'])

# %%
def prompt(item):
    show_tests = lambda x: '\n'.join(x)
    return f"{item['text']} Your code should satisfy the following tests.\n\n{item['test_setup_code']}\n\n{show_tests(item['test_list'])}" if item['test_setup_code']\
        else f"{item['text']} Your code should satisfy the following tests.\n\n{show_tests(item['test_list'])}"

def generate(item, pipe, max_new_tokens=10):
    messages = [{'role': 'user', 'content': prompt(item)}]
    resp = pipe(messages, max_new_tokens=max_new_tokens)[0]['generated_text'][-1]
    return resp['content']
