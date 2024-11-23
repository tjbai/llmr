import json
import yaml
import argparse

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import pipeline

def format_prompt(item):
    show_tests = lambda x: '\n'.join(x)
    return f"{item['text']} Your code should satisfy the following tests.\n\n{item['test_setup_code']}\n\n{show_tests(item['test_list'])}" if item['test_setup_code']\
        else f"{item['text']} Your code should satisfy the following tests.\n\n{show_tests(item['test_list'])}"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(args.config)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    dataset = load_dataset('google-research-datasets/mbpp')
    pipe = pipeline('text-generation', config['model_name'], device=config['device'])
    generate_kwargs = {'num_return_sequences': config['M'], 'temperature': config['temperature'], 'do_sample': True}

    with open(config['output'], 'w') as f:
        for item in tqdm(dataset[config['split']]):
            messages = [{'role': 'user', 'content': format_prompt(item)}]
            gen = pipe(messages, max_new_tokens=config['max_new_tokens'], **generate_kwargs)
            resps = [x['generated_text'][-1]['content'] for x in gen]
            json.dump({'item': item, 'resps': resps}, f)

if __name__ == '__main__':
    main()
