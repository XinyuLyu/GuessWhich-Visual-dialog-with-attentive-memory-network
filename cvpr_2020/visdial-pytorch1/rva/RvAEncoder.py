import torch
from torch import nn
from torch.nn import functional as F

from rva.dynamic import DynamicRNN
from rva.Q_ATT import Q_ATT
from rva.RVA import RvA_MODULE


class RvAEncoder(nn.Module):
    def __init__(self,
                 vocabSize,
                 embedSize,
                 rnnHiddenSize,
                 numLayers,
                 useIm,
                 imgEmbedSize,
                 imgFeatureSize,
                 numRounds,
                 isAnswerer,
                 dropout=0,
                 startToken=None,
                 endToken=None,
                 **kwargs):
        super().__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.word_embed = nn.Embedding(
            vocabSize,
            300,#config["word_embedding_size"],
            padding_idx=0
        )

        self.ques_rnn = nn.LSTM(
            300,#config["word_embedding_size"],
            512,#config["lstm_hidden_size"],
            2,#config["lstm_num_layers"],
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly
        self.ques_rnn = DynamicRNN(self.ques_rnn)

        # self attention for question
        self.Q_ATT_ref = Q_ATT()

        # modules
        self.RvA_MODULE = RvA_MODULE()

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, batch):
        # img - shape: (batch_size, num_proposals, img_feature_size) - RCNN bottom-up features
        img = batch["img_feat"]

        # init language embedding
        # ques_word_embed - shape: (batch_size, num_rounds, quen_len_max, word_embedding_size)
        # ques_word_encoded - shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size)
        # ques_not_pad - shape: (batch_size, num_rounds, quen_len_max)
        # ques_encoded - shape: (batch_size, num_rounds, lstm_hidden_size)
        ques_word_embed, ques_word_encoded, ques_not_pad = self.init_q_embed(batch)

        # question feature for RvA
        # ques_ref_feat - shape: (batch_size, num_rounds, lstm_hidden_size)
        # ques_ref_att - shape: (batch_size, num_rounds, quen_len_max)
        ques_ref_feat, ques_ref_att = self.Q_ATT_ref(ques_word_embed, ques_word_encoded, ques_not_pad)

        # RvA module
        ques_feat = ques_ref_feat
        # img_att - shape: (batch_size, num_rounds, num_proposals)
        img_att, att_set = self.RvA_MODULE(img, ques_feat)
        # img_feat - shape: (batch_size, num_rounds, img_feature_size)
        img_feat = torch.bmm(img_att, img)

        return img_feat

    def init_q_embed(self, batch):
        ques = batch["ques"]  # shape: (batch_size, num_rounds, quen_len_max)
        batch_size, num_rounds, _ = ques.size()

        # question feature
        ques_not_pad = (ques != 0).float()  # shape: (batch_size, num_rounds, quen_len_max)
        ques = ques.view(-1, ques.size(-1))  # shape: (batch_size*num_rounds, quen_len_max)
        ques_word_embed = self.word_embed(ques)  # shape: (batch_size*num_rounds, quen_len_max, lstm_hidden_size)
        ques_word_encoded, _ = self.ques_rnn(ques_word_embed, batch[
            'ques_len'])  # shape: (batch_size*num_rounds, quen_len_max, lstm_hidden_size*2)
        quen_len_max = ques_word_encoded.size(1)
        ques_word_encoded = ques_word_encoded.view(-1, num_rounds, quen_len_max, ques_word_encoded.size(
            -1))  # shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size)
        ques_word_embed = ques_word_embed.view(-1, num_rounds, quen_len_max, ques_word_embed.size(
            -1))  # shape: (batch_size, num_rounds, quen_len_max, word_embedding_size)

        return ques_word_embed, ques_word_encoded, ques_not_pad
