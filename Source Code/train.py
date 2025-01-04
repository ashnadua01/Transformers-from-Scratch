import torch
from torch import nn
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

import nltk
from tqdm import tqdm
from torch.utils.data import DataLoader

from preprocess import read_data, preprocess, build_vocab, tokenize
from utils import TranslationDataset
from transformer import Transformer
from calculate_bleu_score import evaluate_model
import json

nltk.download('punkt')
mp.set_start_method('spawn', force=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Load data
train_en = read_data('/Users/ashnadua/Desktop/2021101072_assignment2/Data/train.en')
train_fr = read_data('/Users/ashnadua/Desktop/2021101072_assignment2/Data/train.fr')
dev_en = read_data('/Users/ashnadua/Desktop/2021101072_assignment2/Data/dev.en')
dev_fr = read_data('/Users/ashnadua/Desktop/2021101072_assignment2/Data/dev.fr')
test_en = read_data('/Users/ashnadua/Desktop/2021101072_assignment2/Data/test.en')
test_fr = read_data('/Users/ashnadua/Desktop/2021101072_assignment2/Data/test.fr')


## Preprocess data
train_en_clean = preprocess(train_en)
train_fr_clean = preprocess(train_fr)
dev_en_clean = preprocess(dev_en)
dev_fr_clean = preprocess(dev_fr)
test_en_clean = preprocess(test_en)
test_fr_clean = preprocess(test_fr)


## Build Vocabulary with minimum frequency = 3
word_to_idx_en, idx_to_word_en = build_vocab(train_en_clean, min_freq=3)
word_to_idx_fr, idx_to_word_fr = build_vocab(train_fr_clean, min_freq=3)

print(f"English Vocab Size: {len(word_to_idx_en)}")
print(f"French Vocab Size: {len(word_to_idx_fr)}")


## Tokenize data
train_en_tokenized = tokenize(train_en_clean)
train_fr_tokenized = tokenize(train_fr_clean)
dev_en_tokenized = tokenize(dev_en_clean)
dev_fr_tokenized = tokenize(dev_fr_clean)
test_en_tokenized = tokenize(test_en_clean)
test_fr_tokenized = tokenize(test_fr_clean)


## Create Datasets and DataLoaders
train_dataset = TranslationDataset(train_en_tokenized, train_fr_tokenized, word_to_idx_en, word_to_idx_fr, max_length=100)
dev_dataset = TranslationDataset(dev_en_tokenized, dev_fr_tokenized, word_to_idx_en, word_to_idx_fr, max_length=100)
test_dataset = TranslationDataset(test_en_tokenized, test_fr_tokenized, word_to_idx_en, word_to_idx_fr, max_length=100)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

en_vocab_size = len(word_to_idx_en)
fr_vocab_size = len(word_to_idx_fr)


## Initialize model, criterion and optimizer
d_model = 300
num_heads = 10
num_layers = 2
d_ff = 300
max_seq_length = 100
dropout = 0.1
num_epochs = 10

model = Transformer(en_vocab_size, fr_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx_en['<PAD>'])
optimizer = optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)


## Train model
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    loop = tqdm(train_loader, leave=False)
    for i, (src, tgt) in enumerate(loop):
        src = src.to(device)
        tgt = tgt.to(device)
        
        optimizer.zero_grad()
        output = model(src, tgt[:, :-1])
        loss = criterion(output.contiguous().view(-1, fr_vocab_size), tgt[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
        loop.set_postfix(loss=loss.item())
    
    epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {epoch_loss}")
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for (src, tgt) in dev_loader:
            src = src.to(device)
            tgt = tgt.to(device)
            
            output = model(src, tgt[:, :-1])
            loss = criterion(output.contiguous().view(-1, fr_vocab_size), tgt[:, 1:].contiguous().view(-1))
            
            val_loss += loss.item()
            
    val_loss = val_loss / len(dev_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}")


## Evaluate model
print("Evaluating model on dev set")
dev_loader = DataLoader(dev_dataset, batch_size=1, shuffle=True)
bleu_score_dev = evaluate_model(model, dev_loader, word_to_idx_en['<PAD>'], word_to_idx_en['</s>'])

print("Evaluating model on test set")
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
bleu_score_test = evaluate_model(model, test_loader, word_to_idx_en['<PAD>'], word_to_idx_en['</s>'])

torch.save(model.state_dict(), '/Users/ashnadua/Desktop/2021101072_assignment2/Models/transformer_model.pt')
vocab_data = {
    'word_to_idx_en': word_to_idx_en,
    'word_to_idx_fr': word_to_idx_fr,}

with open('/Users/ashnadua/Desktop/2021101072_assignment2/Models/transformer_vocab.json', 'w', encoding='utf-8') as f:
    json.dump(vocab_data, f, ensure_ascii=False, indent=4)

torch.save(train_dataset, '/Users/ashnadua/Desktop/2021101072_assignment2/Models/train_dataset.pth')
torch.save(dev_dataset, '/Users/ashnadua/Desktop/2021101072_assignment2/Models/dev_dataset.pth')
torch.save(test_dataset, '/Users/ashnadua/Desktop/2021101072_assignment2/Models/test_dataset.pth')