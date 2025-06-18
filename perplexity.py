import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.n_embd = n_embd
        self.n_head = n_head

        self.c_attn = nn.Linear(n_embd, 3*n_embd)
        self.attn_prediction = nn.Linear(n_embd, n_embd)
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
    def __init__(self, n_embd):
        super().__init__()
        self.fc1 = nn.Linear(n_embd, 4*n_embd)
        self.fc2 = nn.Linear(4*n_embd, n_embd)
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        logits = self.fc2(x)
        return logits


class Block(nn.Module):
    def __init__(self, n_head, n_embd, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mha_layer = MultiHeadAttention(n_embd, n_head, block_size)
        self.ffn_layer = FeedForward(n_embd)
    
    def forward(self, x):
        attn_out = self.mha_layer(self.ln1(x))
        x = x + attn_out
        ffn_out = self.ffn_layer(self.ln2(x))
        x = x + ffn_out
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, n_embd, n_head, vocab_size, block_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        block = [Block(n_head, n_embd) for _ in range(6)]
        self.block = nn.Sequential(*block)
        self.ln = nn.LayerNorm(n_embd)
        self.output_embedding = nn.Linear(n_embd, vocab_size)

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

loss_fn = nn.CrossEntropyLoss()
model = GPTLanguageModel(n_embd=384, n_head=6, vocab_size=65, block_size=128)
model.load_state_dict(torch.load('my_model.pth', weights_only=True))
model.to(device)
model.eval()

val_losses = []
total_correct = 0
model.eval() 
for _ in range(len(val_data) // batch_size): 
    with torch.no_grad():
        x, y = get_batch('val')
        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
        val_losses.append(loss.item())
        predictions = torch.argmax(logits.view(-1, vocab_size), 1)
        total_correct += (predictions == y.view(-1)).sum().item()

avg_val_loss = np.mean(val_losses)
perplexity = np.exp(avg_val_loss)
accuracy = total_correct / ((len(val_data) // batch_size) * batch_size * block_size) 

print(f"Validation Loss: {avg_val_loss:.4f}, Perplexity: {perplexity:.4f}, Validation Accuracy: {accuracy:.4f}")