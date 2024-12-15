from torch import nn
from transformers import DebertaModel, DebertaTokenizer
from torch.utils.data import Dataset, DataLoader

class Router(nn.Module):
    
    def __init__(self, dropout=0.2, hidden_size=768):
        self.bert = DebertaModel.from_pretrained('microsoft/deberta-base')
        self.bert.requires_grad_(False)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        cls = outputs.last_hidden_state[:, 0] # (B, 768)
        logits = self.head(cls)
        return logits
    
    def step(batch):
        pass

def train():
    pass
