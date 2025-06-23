import torch
import wandb
import pickle
import numpy as np
import torch.optim as optim
import GPTLanguageModel from model

# Initial parameters
train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('val.bin', dtype=np.uint16, mode='r')

batch_size = 64
block_size = 128
vocab_size = 65
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data loader
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

model = GPTLanguageModel(n_embd=384, n_head=6, vocab_size=65, block_size=128)
model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
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
            accuracy = total_correct / (200 * batch_size * block_size) 

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