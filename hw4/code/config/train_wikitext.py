out_dir = 'out-wikitext'
eval_interval = 250                 # 每250次迭代进行一次评估
eval_iters = 20                     # 减少计算量，评估时使用20个批次（原为200）
log_interval = 50

always_save_checkpoint = False

wandb_log = False 
wandb_project = 'wikitext_large'
wandb_run_name = 'mini-gpt'

dataset = 'wikitext_large'
gradient_accumulation_steps = 4
batch_size = 12                     # 减少内存占用，调整每批次样本数（原为16）
block_size = 64                     # 缩短序列长度（原为256）

n_layer = 4                         # 减少模型规模（原为8）
n_head = 4                          # 减小注意力头数（原为8）
n_embd = 128                        # 减小嵌入维度（原为512）
dropout = 0.2

learning_rate = 1e-3 
max_iters = 20000
lr_decay_iters = 20000
min_lr = 1e-4 
beta2 = 0.99 

warmup_iters = 100 

