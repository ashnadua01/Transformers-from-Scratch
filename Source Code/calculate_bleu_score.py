import torch
import torch.multiprocessing as mp
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

nltk.download('punkt')
mp.set_start_method('spawn', force=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def calculate_bleu(pred, truth):
    pred = pred.cpu().detach().numpy()
    truth = truth.cpu().detach().numpy()
    
    # Flatten both arrays
    pred = pred.flatten()
    truth = truth.flatten()
    
    # Convert to list of integers
    preds = [int(item) for item in pred]
    truths = [int(item) for item in truth]
    
    # Calculate BLEU score
    chencherry = SmoothingFunction()
    bleu_score = sentence_bleu([truths], preds, smoothing_function=chencherry.method1)
    return bleu_score


def remove_pad_eos(tokens, pad_token_idx, eos_token_idx):
    while tokens and (tokens[-1] == pad_token_idx or tokens[-1] == eos_token_idx):
        tokens.pop()
    return tokens

def evaluate_model(model, test_loader, pad_token_idx, eos_token_idx):
    model.eval()
    model.to(device)

    bleu_scores = []
    sum_bleu_scores = 0
    total_batches = len(test_loader)

    with torch.no_grad():
        for i, (src, tgt) in enumerate(test_loader):
            src = src.to(device)
            tgt = tgt.to(device)
            
            output = model(src, tgt[:, :-1])
            
            predicted_seq = output.argmax(dim=-1) 

            # Remove PAD and EOS tokens from predictions and ground truth
            predicted_seq_list = [remove_pad_eos(pred.tolist(), pad_token_idx, eos_token_idx) for pred in predicted_seq]
            truth_list = [remove_pad_eos(tgt_entry.tolist(), pad_token_idx, eos_token_idx) for tgt_entry in tgt[:, 1:]]

            for pred, truth in zip(predicted_seq_list, truth_list):
                bleu_score = calculate_bleu(torch.tensor(pred), torch.tensor(truth))
                bleu_scores.append(bleu_score)
                sum_bleu_scores += bleu_score
                
#                 print(f'Device: {device}, Batch: {i + 1}, BLEU score: {bleu_score}')
    
    average_bleu_score = sum_bleu_scores / total_batches if total_batches > 0 else 0
    print(f'Average BLEU score over {total_batches} batches: {average_bleu_score}')

    return average_bleu_score