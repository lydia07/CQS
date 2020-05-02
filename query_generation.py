from model import Seq2seq
import os
import torch
import torch.nn.functional as F
import config
import pickle
from preprocess.data_utils import (START_TOKEN, END_TOKEN, UNK_TOKEN, END_ID, get_loader, UNK_ID, word_tokenize,
                                   clean_str, get_spans)
from inference import (Hypothesis, outputids2words)

class QueryGenerator(object):
    def __init__(self, model_path):
        with open(config.word2idx_file, 'rb') as f:
            word2idx = pickle.load(f)

        self.tok2idx = word2idx
        self.idx2tok = {idx: tok for tok, idx in self.tok2idx.items()}
        self.model = Seq2seq(model_path=model_path)
        self.max_length = config.max_seq_len
        self.entity2idx = {
                            'O': 0,
                            'B-keyword': 1,
                            'I-keyword': 2
                            }

    @staticmethod
    def sort_hypotheses(hypotheses):
        return sorted(hypotheses, key=lambda h: h.avg_log_prob, reverse=True)
    
    def beam_search(self, src_seq, ext_src_seq, src_len, tag_seq):
        zeros = torch.zeros_like(src_seq)
        enc_mask = torch.BoolTensor(src_seq == zeros)
        src_len = torch.LongTensor(src_len)
        prev_context = torch.zeros(1, 1, 2 * config.hidden_size)

        if config.use_gpu:
            src_seq = src_seq.to(config.device)
            ext_src_seq = ext_src_seq.to(config.device)
            src_len = src_len.to(config.device)
            enc_mask = enc_mask.to(config.device)
            prev_context = prev_context.to(config.device)

            if config.use_tag:
                tag_seq = tag_seq.to(config.device)
        # forward encoder
        enc_outputs, enc_states = self.model.encoder(src_seq, src_len, tag_seq)
        h, c = enc_states  # [2, b, d] but b = 1
        hypotheses = [Hypothesis(tokens=[self.tok2idx[START_TOKEN]],
                                 log_probs=[0.0],
                                 state=(h[:, 0, :], c[:, 0, :]),
                                 context=prev_context[0]) for _ in range(config.beam_size)]
        # tile enc_outputs, enc_mask for beam search
        ext_src_seq = ext_src_seq.repeat(config.beam_size, 1)
        enc_outputs = enc_outputs.repeat(config.beam_size, 1, 1)
        enc_features = self.model.decoder.get_encoder_features(enc_outputs)
        enc_mask = enc_mask.repeat(config.beam_size, 1)
        num_steps = 0
        results = []
        while num_steps < config.max_decode_step and len(results) < config.beam_size:
            latest_tokens = [h.latest_token for h in hypotheses]
            latest_tokens = [idx if idx < len(self.tok2idx) else UNK_ID for idx in latest_tokens]
            prev_y = torch.LongTensor(latest_tokens).view(-1)

            if config.use_gpu:
                prev_y = prev_y.to(config.device)

            # make batch of which size is beam size
            all_state_h = []
            all_state_c = []
            all_context = []
            for h in hypotheses:
                state_h, state_c = h.state  # [num_layers, d]
                all_state_h.append(state_h)
                all_state_c.append(state_c)
                all_context.append(h.context)

            prev_h = torch.stack(all_state_h, dim=1)  # [num_layers, beam, d]
            prev_c = torch.stack(all_state_c, dim=1)  # [num_layers, beam, d]
            prev_context = torch.stack(all_context, dim=0)
            prev_states = (prev_h, prev_c)
            # [beam_size, |V|]
            logits, states, context_vector = self.model.decoder.decode(prev_y, ext_src_seq,
                                                                       prev_states, prev_context,
                                                                       enc_features, enc_mask)
            h_state, c_state = states
            log_probs = F.log_softmax(logits, dim=1)
            top_k_log_probs, top_k_ids \
                = torch.topk(log_probs, config.beam_size * 2, dim=-1)

            all_hypotheses = []
            num_orig_hypotheses = 1 if num_steps == 0 else len(hypotheses)
            for i in range(num_orig_hypotheses):
                h = hypotheses[i]
                state_i = (h_state[:, i, :], c_state[:, i, :])
                context_i = context_vector[i]
                for j in range(config.beam_size * 2):
                    new_h = h.extend(token=top_k_ids[i][j].item(),
                                     log_prob=top_k_log_probs[i][j].item(),
                                     state=state_i,
                                     context=context_i)
                    all_hypotheses.append(new_h)

            hypotheses = []
            for h in self.sort_hypotheses(all_hypotheses):
                if h.latest_token == END_ID:
                    if num_steps >= config.min_decode_step:
                        results.append(h)
                else:
                    hypotheses.append(h)

                if len(hypotheses) == config.beam_size or len(results) == config.beam_size:
                    break
            num_steps += 1
        if len(results) == 0:
            results = hypotheses
        h_sorted = self.sort_hypotheses(results)

        return h_sorted[0]

    def context2ids(self, tokens, word2idx):
        ids = list()
        extended_ids = list()
        oov_lst = list()
        # START and END token is already in tokens lst
        for token in tokens:
            if token in word2idx:
                ids.append(word2idx[token])
                extended_ids.append(word2idx[token])
            else:
                ids.append(word2idx[UNK_TOKEN])
                if token not in oov_lst:
                    oov_lst.append(token)
                extended_ids.append(len(word2idx) + oov_lst.index(token))
            if len(ids) == self.max_length:
                break
        ids_, extended_ids_ = [], []
        ids_.append(ids)
        extended_ids_.append(extended_ids)
        ids_ = torch.Tensor(ids_).long()
        extended_ids_ = torch.Tensor(extended_ids_).long()

        return ids_, extended_ids_, oov_lst

    def get_tags(self, src_len, span):
        tags = [self.entity2idx['O']] * src_len
        # for  span in spans:
        start = span['start']
        end = span['end']
        for idx in range(start, end + 1):
            if idx == start:
                tags[idx] = self.entity2idx['B-keyword']
            else:
                tags[idx] = self.entity2idx['I-keyword']
        return tags
    
    def get_one_query(self, tokens, tags):
        # add start and end token
        tokens.insert(0, START_TOKEN)
        tokens.append(END_TOKEN)
        tags.insert(0, self.entity2idx['O'])
        tags.append(self.entity2idx['O'])

        src_seq, ext_src_seq, oov_lst = self.context2ids(tokens, self.tok2idx)
        src_len = len(tokens)
        tags_ = []
        tags_.append(tags)
        tag_seq = torch.Tensor(tags_).long()
        oov_lst = (oov_lst)
        src_len_ = []
        src_len_.append(src_len)
        # beam search
        best_question = self.beam_search(src_seq, ext_src_seq, src_len_, tag_seq)
        # discard START token
        output_indices = [int(idx) for idx in best_question.tokens[1:-1]]
        decoded_words = outputids2words(output_indices, self.idx2tok, oov_lst[0])
        try:
            fst_stop_idx = decoded_words.index(END_ID)
            decoded_words = decoded_words[:fst_stop_idx]
        except ValueError:
            decoded_words = decoded_words
        decoded_words = " ".join(decoded_words)
        # print(decoded_words)
        return decoded_words

    def generate_querys(self, paragraph, keyword):
        querys = []
        tokens, tags = [], []
        tokens = word_tokenize(clean_str(paragraph))

        keyword = clean_str(keyword)
        spans = get_spans(tokens, keyword)
        for span in spans:
            tags = self.get_tags(len(tokens), span)
            query = self.get_one_query(tokens, tags)
            querys.append(query)
        return querys


if __name__ == "__main__":
    qg = QueryGenerator(config.model_path)
    para = 'In September 1760, and before any hostilities erupted, \
            Governor Vaudreuil negotiated from Montreal a capitulation \
            with General Amherst. Amherst granted Vaudreuil\'s request \
            that any French residents who chose to remain in the colony \
            would be given freedom to continue worshiping in their Roman \
            Catholic tradition, continued ownership of their property, \
            and the right to remain undisturbed in their homes. The British \
            provided medical treatment for the sick and wounded French soldiers \
            and French regular troops were returned to France aboard British \
            ships with an agreement that they were not to serve again in the \
            present war.'
    keyword = 'French regular troops'
    res = qg.generate_querys(para, keyword)
    print(res)