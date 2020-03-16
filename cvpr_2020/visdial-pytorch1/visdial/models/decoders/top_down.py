import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence

def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths.tolist(), batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class AttModel(nn.Module):
    def __init__(self):
        super(AttModel, self).__init__()
        self.vocab_size = 10157
        self.input_encoding_size = 300
        self.rnn_size = 512
        self.num_layers = 2
        self.drop_prob_lm = 0.5
        self.fc_feat_size = 2048
        self.att_feat_size = 2048
        self.att_hid_size = 512

        self.use_bn = 0

        self.ss_prob = 0.0  # Schedule sampling probability

        self.embed = nn.Sequential(nn.Embedding(self.vocab_size + 2, self.input_encoding_size),
                                   nn.ReLU(),
                                   nn.Dropout(self.drop_prob_lm))
        self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                      nn.ReLU(),
                                      nn.Dropout(self.drop_prob_lm))
        self.att_embed = nn.Sequential(*(
                ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                (nn.Linear(self.att_feat_size, self.rnn_size),
                 nn.ReLU(),
                 nn.Dropout(self.drop_prob_lm))))
        self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)

        self.decoding_constraint = 0

        self.core = TopDownCore()

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_()), #h
                Variable(weight.new(self.num_layers, bsz, self.rnn_size).zero_())) #c

    def forward(self, fc_feats, att_feats, att_masks, seq, quesState):

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        outputs = []

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0:  # otherwiste no need to sample
                sample_prob = fc_feats.data.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[-1].data)  # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
                    it = Variable(it, requires_grad=False)
            else:
                it = seq[:, i].clone()
                # break if all the sequences end
            if i >= 1 and seq[:, i].data.sum() == 0:
                break

            xt = self.embed(it)

            output, state = self.core(xt, fc_feats, att_feats, p_att_feats, att_masks, state, quesState)
            output = F.log_softmax(self.logit(output))
            outputs.append(output)

        output = torch.cat([_.unsqueeze(1) for _ in outputs], 1).contiguous()
        return output

class TopDownCore(nn.Module):
    def __init__(self):
        super(TopDownCore, self).__init__()
        self.drop_prob_lm = 0.5
        self.att_lstm = nn.LSTMCell(300 + 512 * 3, 512) # we, fc, h^2_t-1
        self.lang_lstm = nn.LSTMCell(512 * 2, 512) # h^1_t, \hat v
        self.attention = Attention()

    def forward(self, xt, fc_feats, att_feats, p_att_feats, att_masks, state, quesState):
        prev_h = state[0][-1]
        att_lstm_input = torch.cat([prev_h, fc_feats, xt, quesState], 1)

        h_att, c_att = self.att_lstm(att_lstm_input, (state[0][0], state[1][0]))

        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        lang_lstm_input = torch.cat([att, h_att], 1)

        h_lang, c_lang = self.lang_lstm(lang_lstm_input, (state[0][1], state[1][1]))

        output = F.dropout(h_lang, self.drop_prob_lm, self.training)
        state = (torch.stack([h_att, h_lang]), torch.stack([c_att, c_lang]))

        return output, state


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.rnn_size = 512
        self.att_hid_size = 512

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)

        att_h = self.h2att(h)  # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)  # batch * att_size * att_hid_size
        dot = att + att_h  # batch * att_size * att_hid_size
        dot = F.tanh(dot)  # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)  # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)  # (batch * att_size) * 1
        dot = dot.view(-1, att_size)  # batch * att_size

        weight = F.softmax(dot)  # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True)  # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size)  # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1)  # batch * att_feat_size

        return att_res