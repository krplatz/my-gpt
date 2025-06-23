import torch
import numpy as np
import bitsandbytes as bnb
from model import GPTLanguageModel

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