import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import ModelConfig, GPT

# ----------------------------- Configuration ----------------------------------
# Default values for GPT-2 (124M) training on WikiText
# I/O
out_dir = 'out-wikitext'
eval_interval = 250
log_interval = 10
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch' 

# Data
dataset = 'wikitext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

# Model Architecture
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.2
bias = True

# Optimizer Settings
learning_rate = 6e-4
max_iters = 20000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning Rate Schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 20000
min_lr = 6e-5

# System
device = 'cpu' # you can set device to 'cuda' if you are using a gpu
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True

# Override config from CLI/config file
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}

# ----------------------------- Initialization ----------------------------------
seed_offset = 0

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(42 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# ----------------------------- Dataset Loader ----------------------------------
data_dir = os.path.join('data', dataset)

def get_batch(split):
    data_path = os.path.join(data_dir, f'{split}.bin')
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)

# ----------------------------- Model Initialization ----------------------------
iter_num = 0
best_val_loss = 1e9
meta_path = os.path.join(data_dir, 'meta.pkl')

model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=50304, dropout=dropout)

if init_from == 'scratch':
    print("Initializing a new model from scratch")
    gptconf = ModelConfig(**model_args)
    model = GPT(gptconf)

elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device)
    for k in model_args:
        if k in checkpoint['model_args']:
            model_args[k] = checkpoint['model_args'][k]
    model = GPT(ModelConfig(**model_args))
    state_dict = checkpoint['model']
    for k in list(state_dict):
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size

model.to(device)
raw_model = model

scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
# you could use mixed precision training if you are familar with it
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None

if compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# ----------------------------- Evaluation and Training --------------------------
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

X, Y = get_batch('train')
t0 = time.time()
local_iter_num = 0

while iter_num <= max_iters:
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                print(f"saving checkpoint to {out_dir}")
                torch.save({
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break

    # TODO: Implement the gradient accumulation process
    # accumulate the gradient with several forward-backward process, then call optimizer.step() to update model parameters
    #####################################################
    
    for micro_step in range(gradient_accumulation_steps):
        model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
        logits, loss = model(X, Y)
        # Scale the loss by the number of gradient accumulation steps
        loss = loss / gradient_accumulation_steps
        # Fetch the next training batch
        X, Y = get_batch('train')
        # Perform backpropagation
        scaler.scale(loss).backward()
    if grad_clip != 0.0: 
        scaler.unscale_(optimizer)
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer) 
    # update
    scaler.update()
    # Reset gradients
    optimizer.zero_grad(set_to_none=True)
    
    # ###################################

    t1 = time.time()
    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:
            print(f"iter {iter_num}: loss {lossf:.4f}, time {(t1 - t0)*1000:.2f}ms")
    t0 = t1

    iter_num += 1
    local_iter_num += 1
