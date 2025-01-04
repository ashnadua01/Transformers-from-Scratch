import re
import math
import copy
import torch
from torch import nn
import torch.nn as nn
import torch.optim as optim
from collections import Counter
import torch.nn.functional as F
import torch.multiprocessing as mp

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

nltk.download('punkt')
mp.set_start_method('spawn', force=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def read_data(path):
    with open(path, "r", encoding="utf-8") as f:
        data = [line.strip() for line in f.readlines()]
    return data

def preprocess_sentence(sentence):
    sentence = sentence.lower()
    allowed_characters = r"a-zA-Zà-ÿÀ-Ÿ"

    sentence = re.sub(r"\d+", "<NUM>", sentence)
    sentence = sentence.replace("\n", " ")
    sentence = sentence.replace("\'", "'")
    
    sentence = re.sub(rf"[^ {allowed_characters}\s<NUM>]", "", sentence)
    sentence = re.sub(r" +", " ", sentence)
    
    return sentence.strip()

def preprocess(sentences):
    return [preprocess_sentence(sentence) for sentence in sentences]

def build_vocab(sentences, min_freq=3):
    word_to_idx = {
        '<PAD>': 0,
        '<UNK>': 1,
        '<s>': 2,
        '</s>': 3
    }
    
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    word_counter = Counter()
    for sentence in sentences:
        word_counter.update(sentence.split())
        
    for word, freq in word_counter.items():
        if freq >= min_freq:
            idx = len(word_to_idx)
            word_to_idx[word] = idx
            idx_to_word[idx] = word
            
    return word_to_idx, idx_to_word

def tokenize(sentences):
    return [nltk.word_tokenize(sentence) for sentence in sentences]