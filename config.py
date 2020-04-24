#preprocess
ratio = 0.5
# train file
train_src_file = './dataset/SQuAD_t/para-train.txt'
train_tgt_file = './dataset/SQuAD_t/tgt-train.txt'
# dev file
dev_src_file = './dataset/SQuAD_t/para-dev.txt'
dev_tgt_file = './dataset/SQuAD_t/tgt-dev.txt'
# test file
test_src_file = "./dataset/SQuAD_t/para-test.txt"
test_tgt_file = "./dataset/SQuAD_t/tgt-test.txt"
# embedding and dictionary file
embedding = './dataset/SQuAD_t/embedding.pkl'
word2idx_file = './dataset/SQuAD_t/word2idx.pkl'

# model
vocab_size = 45000
model_path = './save/seq2seq/train_0424003904/20_2.54'
train = False
device = 'cuda:1'
use_gpu = True
# debug = False
freeze_embedding = True

num_epochs = 20
max_seq_len = 400
num_layers = 2
hidden_size = 300
embedding_size = 300
lr = 0.1
batch_size = 64
dropout = 0.3
max_grad_norm = 5.0

use_tag = True
use_pointer = True
beam_size = 10
min_decode_step = 8
max_decode_step = 30
output_dir = "./result/pointer_maxout_ans"