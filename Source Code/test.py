import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from transformer import Transformer
from calculate_bleu_score import evaluate_model
import json
from utils import TranslationDataset
mp.set_start_method('spawn', force=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Load data
train_dataset = torch.load('/Users/ashnadua/Desktop/2021101072_assignment2/Models/train_dataset.pth')
dev_dataset = torch.load('/Users/ashnadua/Desktop/2021101072_assignment2/Models/dev_dataset.pth')
test_dataset = torch.load('/Users/ashnadua/Desktop/2021101072_assignment2/Models/test_dataset.pth')

with open('/Users/ashnadua/Desktop/2021101072_assignment2/Models/transformer_vocab.json', 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)
    word_to_idx_en = vocab_data['word_to_idx_en']
    word_to_idx_fr = vocab_data['word_to_idx_fr']

dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

en_vocab_size = len(word_to_idx_en)
fr_vocab_size = len(word_to_idx_fr)


## Load model
d_model = 300
num_heads = 5
num_layers = 3
d_ff = 300
max_seq_length = 100
dropout = 0.3
num_epochs = 10

model = Transformer(en_vocab_size, fr_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)
model.load_state_dict(torch.load('/Users/ashnadua/Desktop/2021101072_assignment2/Models/transformer_model.pt', map_location=device))
model.eval()


## Evaluate model on dev and test set
print("Evaluating model on dev set")
dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=True)
bleu_score_dev = evaluate_model(model, dev_loader, word_to_idx_en['<PAD>'], word_to_idx_en['</s>'])

print("\n")

print("Evaluating model on test set")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
bleu_score_test = evaluate_model(model, test_loader, word_to_idx_en['<PAD>'], word_to_idx_en['</s>'])