import math
import torch
from torch import nn
import torch.nn as nn
import torch.multiprocessing as mp
import nltk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

nltk.download('punkt')
mp.set_start_method('spawn', force=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TranslationDataset(Dataset):
    def __init__(self, src_sentences, tgt_sentences, word_to_idx_src, word_to_idx_tgt, max_length=100):
        self.src_data = self.convert_tokens_to_idx(src_sentences, word_to_idx_src, max_length)
        self.tgt_data = self.convert_tokens_to_idx(tgt_sentences, word_to_idx_tgt, max_length)
        
    def convert_tokens_to_idx(self, sentences, word_to_idx, max_length):
        indexed_sentences = []
        for tokens in sentences:
            indices = [word_to_idx.get(token, word_to_idx["<UNK>"]) for token in tokens]
            indices = [word_to_idx["<s>"]] + indices[:max_length - 2] + [word_to_idx["</s>"]]
            
            if len(indices) > max_length:
                indices = indices[:max_length]
            
            indexed_sentences.append(torch.tensor(indices))
        return pad_sequence(indexed_sentences, batch_first=True, padding_value=word_to_idx["<PAD>"])
    
    def __len__(self):
        return len(self.src_data)

    def __getitem__(self, idx):
        return self.src_data[idx], self.tgt_data[idx]
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_seq_length, d_model, device=device)
        position = torch.arange(0, max_seq_length, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.pow(10_000, (-torch.arange(0, d_model, 2, device=device).float() / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        return self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        return x.reshape(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))

        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
    

