import json
import yaml
import argparse

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, pipeline

INSTRUCTIONS = 'Your code should satisfy the following tests. Aim for a concise and clean solution.'

def format_prompt(item):
    show_tests = lambda x: '\n'.join(x)
    return f"{item['text']} {INSTRUCTIONS}\n\n{item['test_setup_code']}\n\n{show_tests(item['test_list'])}" if item['test_setup_code']\
        else f"{item['text']} {INSTRUCTIONS}\n\n{show_tests(item['test_list'])}"

def parse_response(tok_ids, tokenizer, last='<|end_header_id|>'):
    resp = tokenizer.decode(tok_ids)
    ind = resp.rindex(last)
    return resp[ind+len(last):]

def load_batches(items, B):
    for i in range(0, len(items), B):
        yield items[i:i + B]

def process(batch, pipe, tokenizer, pipe_kwargs):
    messages = [{'role': 'user', 'content': format_prompt(item)} for item in batch]
    gens = pipe(messages, **pipe_kwargs)

    results = []
    for i, item in enumerate(batch):
        s = i * pipe_kwargs['num_return_sequences']
        e = s + pipe_kwargs['num_return_sequences']
        item_gens = gens[s:e]

        tok_ids = [x['generated_token_ids'] for x in item_gens]
        resps = [parse_response(resp, tokenizer) for resp in tok_ids]
        results.append({'item': item, 'resps': resps, 'num_toks': list(map(len, tok_ids))})

    return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--output')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = load_dataset('google-research-datasets/mbpp')
    pipe = pipeline('text-generation', config['model_name'], device=config['device'])
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])

    pipe_kwargs = {
        'max_new_tokens': config['max_new_tokens'],
        'return_tensors': True,
        'num_return_sequences': config['M'],
        'temperature': config['temperature'],
        'do_sample': True,
        'pad_token_id': tokenizer.eos_token_id
    }

    with open(config['output'], 'w') as f:
        for batch in tqdm(load_batches(dataset[config['split']], config['B'])):
            results = process(batch, pipe, tokenizer, pipe_kwargs)
            for result in results:
                json.dump(result, f)
                f.write('\n')

if __name__ == '__main__':
    main()
