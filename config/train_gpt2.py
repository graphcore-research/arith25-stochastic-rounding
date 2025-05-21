# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

wandb_log = True
wandb_project = "arith-sr"
wandb_run_name = "gpt2d"

n_layer = 24
n_head = 16
n_embd = 1024

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520
batch_size = 12
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 30000

dtype = "bfloat16"
qat_start_iter = 100

decay_lr = False
# lr_decay_iters = 60000
learning_rate = 3e-4  # max learning rate
# min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# eval stuff
eval_interval = 500
eval_iters = 200
log_interval = 20

# weight decay
weight_decay = 1e-1
