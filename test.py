# %%
%load_ext autoreload
%autoreload 2

# %%
from datasets import load_dataset
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
resp = pipe(messages, max_new_tokens=10, return_tensors=True, **generate_kwargs)

# %%
code = resp[0]['generated_text'][-1]['content']

# %%
print(dataset['train'][0]['text'])
print(dataset['train'][0]['test_list'])

# %%
test_code = '''
import sys
print("Hello from stdout")
print("Error message", file=sys.stderr)
while True:
    continue
'''

from eval import exec_with_timeout, eval_tests

# exec_with_timeout(test_code, 10)
eval_tests(test_code, ['print(\'hello world\')'], timeout=10)
# %%
import json

# happy claude-generated test cases
sample_data = [
    {
        "item": {
            "task_id": "1",
            "text": "Write a function to calculate factorial",
            "test_list": [
                "assert factorial(5) == 120",
                "assert factorial(0) == 1",
                "assert factorial(3) == 6"
            ],
        },
        "resps": [
            """
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
print("Calculating factorial...")
            """,

            """
def factorial(n)
    return n * factorial(n-1)
            """,

            """
import sys
def factorial(n):
    print("Debug info", file=sys.stderr)
    if n < 0:
        raise ValueError("n must be >= 0")
    if n == 0:
        return 1
    return n * factorial(n-1)
            """,

            """
def factorial(n):
    while True:
        pass
    return 1
            """,

            """
def factorial(n):
    if n == 0:
        return 1
    return n * (n-1)  # Wrong implementation
            """
        ]
    },
    {
        "item": {
            "task_id": "2",
            "text": "Write a function to check if a number is prime",
            "test_list": [
                "assert is_prime(2) == True",
                "assert is_prime(4) == False",
                "assert is_prime(17) == True"
            ],
        },
        "resps": [
            """
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
            """,

            """
def is_prime(n):
    print(f"Checking if {n} is prime")
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
            """
        ]
    }
]

with open('test_exec.jsonl', 'w') as f:
    for item in sample_data:
        json.dump(item, f)
        f.write('\n')
