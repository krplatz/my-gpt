import torch
import wandb
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
    def __init__(self, n_embd, n_head):
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
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mha_layer = MultiHeadAttention(n_embd, n_head)
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

model = GPTLanguageModel(n_embd=384, n_head=6, vocab_size=65, block_size=128)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
hyperparameters = {'epochs': 5, 'lr': 0.001, 'n_embd': 384, 'n_head': 6, 'vocab_size': 65, 'batch_size': 64, 'block_size': 128}
wandb.init(config=hyperparameters)
epochs = 1

for epoch in range(epochs):
    total_loss = 0
    print(f"Epoch: {epoch}/{epochs}")

    for batch in range(len(train_data) // batch_size):
        x, y = get_batch('train')
        optimizer.zero_grad()
        logits = model(x)
        logits = logits.view(-1, vocab_size)
        next_char = y.view(-1,)
        loss = loss_fn(logits, next_char)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        loss_value = total_loss / (len(train_data) // batch_size)

        if batch % 1000 == 0:
            val_losses = []
            total_correct = 0
            model.eval() 
            for _ in range(200): 
                with torch.no_grad():
                    x, y = get_batch('val')
                    logits = model(x)
                    loss = loss_fn(logits.view(-1, vocab_size), y.view(-1))
                    val_losses.append(loss.item())
                    predictions = torch.argmax(logits.view(-1, vocab_size), 1)
                    total_correct += (predictions == y.view(-1)).sum().item()
    
            model.train() 
            
            avg_val_loss = np.mean(val_losses)
            accuracy = total_correct / (200 * block_size) 

            print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {accuracy:.4f}")
            log_value = {'train_loss': loss.item(), 'val_loss': avg_val_loss, 'val_accuracy': accuracy}
            wandb.log(log_value)

torch.save(model.state_dict(), 'my_model.pth')

with open('meta.pkl','rb') as f:
    meta = pickle.load(f)

def generate(model, start_prompt, max_new_tokens):
    model.eval()
    encoded_prompt = []
    decoded_prompt = []
    for char in start_prompt:
        stoi_dict = meta['stoi']
        char_id = stoi_dict[char]
        encoded_prompt.append(char_id)
    
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([encoded_prompt]).to(device))
        last_logit = logits[:, -1, :]
        predictions = torch.argmax(last_logit, 1)
        encoded_prompt.append(predictions.item())

    for i in encoded_prompt:
        itos_dict = meta['itos']
        i_id = itos_dict[i]
        decoded_prompt.append(i_id)
    
    return "".join(decoded_prompt)

print("Training finished. Here is some generated text:")
generated_text = generate(model, start_prompt="\n", max_new_tokens=100)
print(generated_text)