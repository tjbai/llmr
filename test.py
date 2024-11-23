# %%
from datasets import load_dataset

# %%
dataset = load_dataset('google-research-datasets/mbpp')

# %%
import torch
from transformers import pipeline
pipe = pipeline('text-generation', 'meta-llama/Llama-3.2-1B-Instruct', torch_dtype=torch.float16)

# %%
def prompt(item):
    show_tests = lambda x: '\n'.join(x)
    return f"{item['text']} Your code should satisfy the following tests.\n\n{item['test_setup_code']}\n\n{show_tests(item['test_list'])}" if item['test_setup_code']\
        else f"{item['text']} Your code should satisfy the following tests. Avoid including comments.\n\n{show_tests(item['test_list'])}"

# %%
messages = [{'role': 'user', 'content': prompt(dataset['train'][0])}]
generate_kwargs = {'num_return_sequences': 1, 'do_sample': True, 'temperature': 0.7}
resp = pipe(messages, max_new_tokens=128, **generate_kwargs)

# %%
code = resp[0]['generated_text'][-1]['content']
print(code)

# %%
print(dataset['train'][0]['text'])
print(dataset['train'][0]['test_list'])

# %%
exec(code)
