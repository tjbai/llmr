import json
import yaml
import argparse
import random
from tqdm import tqdm

import wandb
import torch
import torch.nn.functional as F
import numpy as np
import nlpaug.augmenter.word as naw

from torch import nn
from torch.optim import AdamW
from transformers import DebertaV2Model, DebertaV2TokenizerFast
from torch.utils.data import Dataset, DataLoader, random_split
from ema_pytorch import EMA

import nltk
nltk.download('averaged_perceptron_tagger_eng')

class RouterDataset(Dataset):
    SEED = 42

    def __init__(self, input, augment=False, aug_factor=3):
        random.seed(self.SEED)
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        if torch.cuda.is_available(): torch.cuda.manual_seed(self.SEED) 
        
        self.tokenizer = DebertaV2TokenizerFast.from_pretrained('microsoft/deberta-v3-small')
        self.augmenters = [naw.SynonymAug(), naw.ContextualWordEmbsAug()]
    
        with open(input, 'r') as f:
            self.items = [json.loads(line) for line in f]
        
        if augment:
            aug_items = []
            print(f'generating items with aug_factor={aug_factor}')
            for item in tqdm(self.items):
                for _ in range(aug_factor):
                    aug = random.choice(self.augmenters)
                    new_prompt = aug.augment(item['prompt'])[0]
                    aug_items.append({'prompt': new_prompt, 'target': item['target']})
            self.items.extend(aug_items)
        
        self.tokenized_items = [] 
        for item in self.items:
            inputs = self.tokenizer(
                item['prompt'],
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            self.tokenized_items.append({
                'input_ids': inputs['input_ids'].squeeze(),
                'attention_mask': inputs['attention_mask'].squeeze(),
                'target': item['target'],
            })

    def __getitem__(self, idx):
        return self.tokenized_items[idx]
    
    def __len__(self):
        return len(self.tokenized_items)

class Router(nn.Module):
    def __init__(self, dropout=0.25, hidden_size=384):
        super().__init__()
        self.bert = DebertaV2Model.from_pretrained('microsoft/deberta-v3-small')
        self.bert.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(768, hidden_size),
            nn.Dropout(dropout),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = outputs.last_hidden_state[:, 0] # (B, 768)
        logits = self.head(cls).squeeze()
        return logits
    
    def step(self, batch):
        logits = self.forward(batch['input_ids'], batch['attention_mask'])
        return F.binary_cross_entropy_with_logits(logits, batch['target'])
    
def get_loaders(config):
    dataset = RouterDataset(
        config['data']['input'],
        augment=config['data'].get('augment', False),
        aug_factor=config['data'].get('aug_factor', 3)
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [len(dataset) - config['data']['val_size'], config['data']['val_size']],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    return train_loader, val_loader

def compute_ece(preds, targets, n_bins=10):
    ece = 0
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        bin_mask = (preds >= bin_boundaries[i]) & (preds < bin_boundaries[i+1])
        if not any(bin_mask): continue
        bin_conf = preds[bin_mask].mean()
        bin_acc = targets[bin_mask].mean()
        ece += (sum(bin_mask) / len(preds)) * abs(bin_acc - bin_conf)
    return ece

@torch.no_grad()
def val(model, val_loader, config, max_steps=None):
    model.eval()
    tot_loss = N = 0
    all_preds = []
    all_targets = []
    
    for i, batch in enumerate(val_loader):
        if max_steps and i >= max_steps: break
        batch = {k: v.to(config['device']) for k, v in batch.items()}
        logits = model(batch['input_ids'], batch['attention_mask'])
        loss = F.binary_cross_entropy_with_logits(logits, batch['target'])
        
        preds = torch.sigmoid(logits)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(batch['target'].cpu().numpy())
        
        tot_loss += loss.item()
        N += 1

    # get calibration scores too
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    ece = compute_ece(all_preds, all_targets)
    brier = np.mean((all_preds - all_targets) ** 2)
    
    wandb.log({'val/loss': tot_loss / N, 'val/ece': ece, 'val/brier': brier})

def train(config):
    device = config['device']
    train_loader, val_loader = get_loaders(config)
    model = Router(dropout=config['model']['dropout'], hidden_size=config['model']['hidden_size']).to(device)
    ema = EMA(model, beta=0.999, update_after_step=20, update_every=1)
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
            ema.update()

            wandb.log({'train/loss': loss})
        
        val(ema, val_loader, config)

        torch.save({
            'model': model.head.state_dict(),
            'ema': {k: v for k, v in ema.state_dict().items() if k.startswith('ema_model.head')},
            'optim': optim.state_dict()
        }, f"{config['checkpoint']}/{config['wandb']['run_name']}_epoch={epoch+1}.pt")
    
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
