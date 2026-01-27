import time

#out_dir = 'out-owt-gpt2mini'
out_dir = 'out-eyereward-all'
eval_interval = 50
eval_iters = 80
wandb_log = False 
wandb_project = 'eyereward'
wandb_run_name = 'eyereward'

dataset = 'eyereward'
init_from = 'resume'

# only save checkpoints if the validation loss improves
always_save_checkpoint = True 

#n_layer = 24
#n_head = 16
#n_embd = 1024
#dropout = 0.2

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters

# cgpt 170k has 37,399,419

# model iters
# 38m parameters model has 757,000 iters
# 100m parameters model has 762,000 iters
# 350m parameters model has 272,000 iters

block_size = 1024

batch_size = 2 #16
gradient_accumulation_steps = 32 

# 13,578,520 tokens for participant 8  

#max_iters = 127000 + 150 * 7  learning_rate = 1e-5
max_iters = 127000 + 15 * 8

# finetune at constant LR
#learning_rate = 3e-7
learning_rate = 5e-6
#decay_lr = True 
