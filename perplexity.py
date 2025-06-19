import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F, copy
import bitsandbytes as bnb

# Initial parameters
train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')

batch_size = 64
block_size = 128
vocab_size = 65
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batch(split):
    if split == 'train':
        data = train_data
    elif split == 'val':
        data = val_data

    ix = np.random.randint(0, len(data)-block_size, (batch_size,))
    offsets = np.arange(block_size)
    indices = ix[:, None] + offsets
    x = torch.from_numpy(data[indices]).long()
    x = x.to(device)
    y = torch.from_numpy(data[indices+1]).long()
    y = y.to(device)

    return x, y

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, Linear=nn.Linear):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head

        self.c_attn = Linear(n_embd, 3*n_embd)
        self.attn_prediction = Linear(n_embd, n_embd)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)) == 0)
    
    def forward(self, x):
        B, T, C = x.shape
        qkv = self.c_attn(x)
        Q, K, V = qkv.split(self.n_embd, dim=2)
        head_size = self.n_embd // self.n_head

        Q = Q.view(B, T, self.n_head, head_size).transpose(1, 2) # (B, n_head, T, head_size)
        K = K.view(B, T, self.n_head, head_size).transpose(1, 2)
        V = V.view(B, T, self.n_head, head_size).transpose(1, 2)

        attn_scores = Q @ K.transpose(-2, -1) # (B, n_head, T, T)
        attn_scores = attn_scores / (head_size**0.5)
        attn_scores = attn_scores.masked_fill(self.mask[:T, :T], float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = attn_weights @ V # (B, n_head, T, head_size)
        attn_output = attn_output.transpose(1, 2).reshape(B, T, self.n_embd)
        attn_prediction = self.attn_prediction(attn_output)
        return attn_prediction

class FeedForward(nn.Module):
    def __init__(self, n_embd, Linear=nn.Linear):
        super().__init__()
        self.fc1 = Linear(n_embd, 4*n_embd)
        self.fc2 = Linear(4*n_embd, n_embd)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        logits = self.fc2(x)
        return logits

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, Linear=nn.Linear):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mha_layer = MultiHeadAttention(n_embd, n_head, block_size, Linear=Linear)
        self.ffn_layer = FeedForward(n_embd)
    
    def forward(self, x):
        attn_out = self.mha_layer(self.ln1(x))
        x = x + attn_out
        ffn_out = self.ffn_layer(self.ln2(x))
        x = x + ffn_out
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, n_embd=384, n_head=6, vocab_size=65, block_size=128, Linear=nn.Linear):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        block = [Block(n_embd, n_head, block_size) for _ in range(6)]
        self.block = nn.Sequential(*block)
        self.ln = nn.LayerNorm(n_embd)
        self.output_embedding = Linear(n_embd, vocab_size)

    def forward(self, x):
        B, T = x.shape
        indices = torch.arange(T).to(device)
        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(indices)
        combined_embedding = token_embedding + position_embedding

        block = self.block(combined_embedding)
        ln = self.ln(block)
        output_embedding = self.output_embedding(ln)
        return output_embedding

# Full precision
fp32_model = GPTLanguageModel().to(device)
fp32_model.load_state_dict(torch.load('my_model.pth', map_location=device))
fp32_model.eval()

# 4-bit quantized
q_model = GPTLanguageModel(Linear=bnb.nn.Linear4bit)
q_model.load_state_dict(fp32_model.state_dict())
torch.save(q_model.state_dict(), 'tinygpt_nf4.pth')
q_model.to(device).eval()

loss_fn = nn.CrossEntropyLoss()

# Perplexity eval
@torch.inference_mode()
def eval_ppl(model):
    tot_loss, tot_tokens = 0.0, 0
    for _ in range(len(val_data) // batch_size):
        x, y   = get_batch('val')
        logits = model(x)
        loss   = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        tot_loss  += loss.item() * x.numel()
        tot_tokens += x.numel()
    return np.exp(tot_loss / tot_tokens)

ppl_fp32 = eval_ppl(fp32_model)
ppl_int4 = eval_ppl(q_model)

print(f'Perplexity (fp32): {ppl_fp32:.3f}')
print(f'Perplexity (int4): {ppl_int4:.3f}')
print(f'Delta PPL: {(ppl_int4 / ppl_fp32 - 1) * 100:.2f}%')