import sys
sys.path.append('../')
from data_utils import (SquadData, make_vocab, make_embedding)
from bert import Ner
# import config

def make_para_dataset():
    embedding_file = "/data/hyx/workspace/glove.6B.300d.txt"
    embedding = '/data/hyx/workspace/CQS/dataset/SQuAD_t/embedding.pkl'
    src_word2idx_file = '/data/hyx/workspace/CQS/dataset/SQuAD_t/word2idx.pkl'

    train_squad = '/data/hyx/workspace/CQS/dataset/SQuAD/train-v2.0.json'
    dev_squad = '/data/hyx/workspace/CQS/dataset/SQuAD/dev-v2.0.json'

    train_src_file = '/data/hyx/workspace/CQS/dataset/SQuAD_t/para-train.txt'
    train_tgt_file = '/data/hyx/workspace/CQS/dataset/SQuAD_t/tgt-train.txt'
    dev_src_file = '/data/hyx/workspace/CQS/dataset/SQuAD_t/para-dev.txt'
    dev_tgt_file = '/data/hyx/workspace/CQS/dataset/SQuAD_t/tgt-dev.txt'

    # preprocess training data
    ner_model = Ner('/data/hyx/workspace/BERT-NER/out_base/')
    train_data = SquadData(ner_model=ner_model, data_dir=train_squad)
    # pks: paragraph with keyword pairs
    train_pks, counter = train_data.process_data()
    train_data.make_tag_format(train_pks, train_src_file, train_tgt_file)
    vocab_size = 45000
    word2idx = make_vocab(counter, src_word2idx_file, vocab_size)
    make_embedding(embedding_file, embedding, word2idx)

    # preprocess dev data
    dev_data = SquadData(ner_model=ner_model, data_dir=dev_squad)
    dev_pks, counter = dev_data.process_data()
    dev_data.make_tag_format(dev_pks, dev_src_file, dev_tgt_file)


if __name__ == "__main__":
    # make_sent_dataset()
    make_para_dataset()
