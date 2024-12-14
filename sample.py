import json
import yaml
import argparse

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline
from huggingface_hub import InferenceClient

INSTRUCTIONS = 'Your code should satisfy the following tests. Aim for a concise and clean solution. Only return code.'

def format_prompt(item):
    show_tests = lambda x: '\n'.join(x)
    return f"{item['text']} {INSTRUCTIONS}\n\n{item['test_setup_code']}\n\n{show_tests(item['test_list'])}" if item['test_setup_code']\
        else f"{item['text']} {INSTRUCTIONS}\n\n{show_tests(item['test_list'])}"

def parse_response(tok_ids, tokenizer, _='<|end_header_id|>'):
    return tokenizer.decode(tok_ids)

# hacky
def load_batches(items, B):
    for i in range(0, len(items), B):
        yield [dict(items[j]) for j in range(i, min(i + B, len(items)))]

def process_local(batch, pipe, tokenizer, config):
    pipe_kwargs = {
        'max_new_tokens': config['max_new_tokens'],
        'return_tensors': True,
        'num_return_sequences': config['M'],
        'temperature': config['temperature'],
        'do_sample': True,
        'pad_token_id': tokenizer.eos_token_id,
        'eos_token_id': 128009
    }

    # also hacky but prefills assistant for consistent formatting
    msgs = [[{'role': 'user', 'content': format_prompt(item)}, {'role': 'assistant', 'content': '```python'}] for item in batch]    
    tokenized_msgs = [tokenizer.decode(tokenizer.apply_chat_template(m)[-1]) for m in msgs]
    gens = pipe(tokenized_msgs, **pipe_kwargs)

    results = []
    for i, item in enumerate(batch):
        tok_ids = [x['generated_token_ids'] for x in gens[i]]
        resps = [parse_response(resp, tokenizer) for resp in tok_ids]
        results.append({'item': item, 'resps': resps, 'num_toks': list(map(len, tok_ids))})

    return results

def process_hf(batch, client, tokenizer, config):
    gen_kwargs = {
        'max_new_tokens': config['max_new_tokens'],
        'temperature': config['temperature'],
        'n': config['M'],
        'do_sample': True
    }

    results = []
    for item in batch:
        msgs = [{'role': 'user', 'content': format_prompt(item)}, {'role': 'assistant', 'content': '```python'}]
        tokenized_msg = [tokenizer.decode(tokenizer.apply_chat_template(m)[:-1]) for m in msgs]
        completions = client.text_generation(tokenized_msg, **gen_kwargs)
        if isinstance(completions, list): resps = completions
        else: resps = [completions]
        results.append({'item': item, 'resps': resps, 'num_toks': [len(resp.split()) for resp in resps]})

    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    dataset = load_dataset('google-research-datasets/mbpp')
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    
    if config.get('use_hf', False):
        client = InferenceClient(api_key=config['hf_api_key'])
        process_fn = lambda batch: process_hf(batch, client, tokenizer, config)
    else:
        model_kwargs = {'torch_dtype': torch.bfloat16}
        pipe = pipeline('text-generation', model=config['model'], model_kwargs=model_kwargs, device=config['device'])
        process_fn = lambda batch: process_local(batch, pipe, tokenizer, config)

    with open(config['output'], 'w') as f:
        ds = dataset[config['split']]
        total = len(ds) // config['B']
        for batch in tqdm(load_batches(ds, config['B']), total=total):
            for result in process_fn(batch):
                json.dump(result, f)
                f.write('\n')
            
            print('early exiting for test purposes')
            break

if __name__ == '__main__':
    main()
