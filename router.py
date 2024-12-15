import json
import yaml
import argparse
from tqdm import tqdm

import wandb
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from transformers import DebertaModel, DebertaV2TokenizerFast
from torch.utils.data import Dataset, DataLoader

class RouterDataset(Dataset):
    def __init__(self, input, t=0):
        self.t = t
        self.tokenizer = DebertaV2TokenizerFast.from_pretrained('microsoft/deberta-v3-small')

        self.items = []
        with open(input, 'r') as f:
            for line in f:
                o = json.loads(line) # {'prompt': ..., 'target': ...}
                inputs = self.tokenizer(
                    o['prompt'],
                    padding='max_length',
                    max_length=512,
                    truncation=True,
                    return_tensors='pt'
                )
                self.items.append({
                    'input_ids': inputs['input_ids'].squeeze(),
                    'attention_mask': inputs['attention_mask'].squeeze(),
                    'target': o['target']
                })

    def __getitem__(self, idx):
        return self.items[idx]
    
    def __len__(self):
        return len(self.items)

class Router(nn.Module):
    def __init__(self, dropout=0.25, hidden_size=384):
        self.bert = DebertaModel.from_pretrained('microsoft/deberta-v3-small')
        self.bert.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = outputs.last_hidden_state[:, 0] # (B, 768)
        logits = self.head(cls)
        return logits
    
    def step(self, batch):
        logits = self.forward(batch['input_ids'], batch['attention_mask'])
        return F.binary_cross_entropy_with_logits(logits, batch['targets'])
    
def train(config):
    device = config['device']

    model = Router(dropout=config['model']['dropout'], hidden_size=config['model']['hidden_size']).to(device)
    train_dataset = RouterDataset(config['data']['train'])
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    optim = AdamW(model.parameters(), lr=config['training']['lr'])
    
    for epoch in range(config['training']['epochs']):
        model.train()
        for batch in tqdm(train_loader, desc=f'epoch {epoch+1}'):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model.step(batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad_norm'])
            optim.zero_grad()
            optim.step()

            wandb.log({'train/loss': loss})
        
        torch.save(
            {'model': model.head.state_dict(), 'optim': optim.state_dict()},
            f"{config['checkpoint']}/{config['model']['name']}_epoch={epoch+1}.pt"
        )
    
    wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f: config = yaml.safe_load(args)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.init(project='llmr', name=config['wandb']['run_name'], config=config)
    train(config)

if __name__ == '__main__':
    main()