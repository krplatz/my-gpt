import torch
import torch.nn as nn
import torch.nn.functional as F

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
        indices = torch.arange(T).to(x.device)
        token_embedding = self.token_embedding(x)
        position_embedding = self.position_embedding(indices)
        combined_embedding = token_embedding + position_embedding

        block = self.block(combined_embedding)
        ln = self.ln(block)
        output_embedding = self.output_embedding(ln)
        return output_embedding