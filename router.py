import json
import yaml
import argparse
from tqdm import tqdm

import wandb
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from transformers import DebertaV2Model, DebertaV2TokenizerFast
from torch.utils.data import Dataset, DataLoader, random_split

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
        super().__init__()
        self.bert = DebertaV2Model.from_pretrained('microsoft/deberta-v3-small')
        self.bert.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )
        # self.head = nn.Sequential(
        #     nn.Linear(768, hidden_size),
        #     nn.Dropout(dropout),
        #     nn.Tanh(),
        #     nn.Linear(hidden_size, 1)
        # )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = outputs.last_hidden_state[:, 0] # (B, 768)
        logits = self.head(cls).squeeze()
        return logits
    
    def step(self, batch):
        logits = self.forward(batch['input_ids'], batch['attention_mask'])
        return F.binary_cross_entropy_with_logits(logits, batch['target'])
    
def get_loaders(config):
    dataset = RouterDataset(config['data']['input'])
    train_dataset, val_dataset = random_split(
        dataset,
        [len(dataset) - config['data']['val_size'], config['data']['val_size']],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    return train_loader, val_loader

@torch.no_grad()
def val(model, val_loader, config, max_steps=None):
    model.eval()
    tot = N = 0
    for i, batch in enumerate(val_loader):
        if max_steps and i >= max_steps: break
        batch = {k: v.to(config['device']) for k, v in batch.items()}
        loss = model.step(batch)
        tot += loss
        N += 1
    return tot / N

def train(config):
    device = config['device']
    train_loader, val_loader = get_loaders(config)
    model = Router(dropout=config['model']['dropout'], hidden_size=config['model']['hidden_size']).to(device)
    optim = AdamW(model.parameters(), lr=config['training']['lr'])
    
    print('sanity check loading...')
    val(model, val_loader, config, 1)
    print('passed!')
    
    for epoch in range(config['training']['epochs']):
        model.train()
        for batch in tqdm(train_loader, desc=f'epoch {epoch+1}'):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model.step(batch)

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['clip_grad_norm'])
            optim.step()

            wandb.log({'train/loss': loss})
        
        wandb.log({'val/loss': val(model, val_loader, config)})
        torch.save({
            'model': model.head.state_dict(), 'optim': optim.state_dict()},
            f"{config['checkpoint']}/{config['model']['name'].split('/')[1]}_epoch={epoch+1}.pt"
        )
    
    wandb.finish()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    return parser.parse_args()

def main():
    args = parse_args()
    with open(args.config) as f: config = yaml.safe_load(f)
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    wandb.init(project='llmr', name=config['wandb']['run_name'], config=config)
    train(config)

if __name__ == '__main__':
    main()
