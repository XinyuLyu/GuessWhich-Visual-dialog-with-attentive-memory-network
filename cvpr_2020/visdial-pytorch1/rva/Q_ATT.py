import torch
from torch import nn
from torch.nn import functional as F


class GatedTrans(nn.Module):
    """docstring for GatedTrans"""

    def __init__(self, in_dim, out_dim):
        super(GatedTrans, self).__init__()

        self.embed_y = nn.Sequential(
            nn.Linear(
                in_dim,
                out_dim
            ),
            nn.Tanh()
        )
        self.embed_g = nn.Sequential(
            nn.Linear(
                in_dim,
                out_dim
            ),
            nn.Sigmoid()
        )

    def forward(self, x_in):
        x_y = self.embed_y(x_in)
        x_g = self.embed_g(x_in)
        x_out = x_y * x_g

        return x_out


class Q_ATT(nn.Module):
    """Self attention module of questions."""

    def __init__(self):
        super(Q_ATT, self).__init__()

        self.embed = nn.Sequential(
            nn.Dropout(p=0.3),#config["dropout_fc"]),
            GatedTrans(
                512*2,#config["lstm_hidden_size"] * 2,
                512,#config["lstm_hidden_size"]
            ),
        )
        self.att = nn.Sequential(
            nn.Dropout(p=0.3),#config["dropout_fc"]),
            nn.Linear(
                512,#config["lstm_hidden_size"],
                1
            )
        )
        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, ques_word, ques_word_encoded, ques_not_pad):
        # ques_word shape: (batch_size, num_rounds, quen_len_max, word_embed_dim)
        # ques_embed shape: (batch_size, num_rounds, quen_len_max, lstm_hidden_size * 2)
        # ques_not_pad shape: (batch_size, num_rounds, quen_len_max)
        # output: img_att (batch_size, num_rounds, embed_dim)

        ques_embed = self.embed(ques_word_encoded)  # shape: (batch_size, num_rounds, quen_len_max, embed_dim)
        ques_norm = F.normalize(ques_embed, p=2, dim=-1)  # shape: (batch_size, num_rounds, quen_len_max, embed_dim)

        att = self.att(ques_norm).squeeze(-1)  # shape: (batch_size, num_rounds, quen_len_max)
        # ignore <pad> word
        att = self.softmax(att)
        att = att * ques_not_pad  # shape: (batch_size, num_rounds, quen_len_max)
        att = att / torch.sum(att, dim=-1, keepdim=True)  # shape: (batch_size, num_rounds, quen_len_max)
        feat = torch.sum(att.unsqueeze(-1) * ques_word, dim=-2)  # shape: (batch_size, num_rounds, rnn_dim)

        return feat, att