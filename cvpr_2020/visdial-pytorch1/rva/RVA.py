import torch
from torch import nn
from torch.nn import functional as F
from rva.Q_ATT import GatedTrans



class ATT_MODULE(nn.Module):
    """docstring for ATT_MODULE"""

    def __init__(self):
        super(ATT_MODULE, self).__init__()

        self.V_embed = nn.Sequential(
            nn.Dropout(p=0.3),#config["dropout_fc"]),
            GatedTrans(
                512,#config["img_feature_size"],
                512#config["lstm_hidden_size"]
            ),
        )
        self.Q_embed = nn.Sequential(
            nn.Dropout(p=0.3),#config["dropout_fc"]),
            GatedTrans(
                300,#config["word_embedding_size"],
                512#config["lstm_hidden_size"]
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

    def forward(self, img, ques):
        # input
        # img - shape: (batch_size, num_proposals, img_feature_size)
        # ques - shape: (batch_size, num_rounds, word_embedding_size) -> (batch_size, word_embedding_size)
        # output
        # att - shape: (batch_size, num_rounds, num_proposals)

        batch_size = ques.size(0)
        num_proposals = img.size(1)

        img_embed = img.view(-1, img.size(-1))  # shape: (batch_size * num_proposals, img_feature_size)
        img_embed = self.V_embed(img_embed)  # shape: (batch_size, num_proposals, lstm_hidden_size)
        img_embed = img_embed.view(batch_size, num_proposals,
                                   img_embed.size(-1))  # shape: (batch_size, num_proposals, lstm_hidden_size)

        ques_embed = ques.unsqueeze(1).repeat(1, num_proposals, 1)  # shape: (batch_size, num_proposals, lstm_hidden_size)
        att_embed = F.normalize(img_embed * ques_embed, p=2, dim=-1)  # (batch_size, num_rounds, num_proposals, lstm_hidden_size)
        att_embed = self.att(att_embed).squeeze(-1)  # (batch_size, num_rounds, num_proposals)
        att = self.softmax(att_embed)  # shape: (batch_size, num_rounds, num_proposals)
        img_feat = torch.bmm(att.view(batch_size,1,num_proposals), img_embed).squeeze(1)
        return img_feat, att


class RvA_MODULE(nn.Module):
    """docstring for R_CALL"""

    def __init__(self):
        super(RvA_MODULE, self).__init__()
        self.ATT_MODULE = ATT_MODULE()

    def forward(self, img, ques):
        # img shape: [batch_size, num_proposals, i_dim]
        # img_att_ques shape: [batch_size, num_rounds, num_proposals]
        # img_att_cap shape: [batch_size, 1, num_proposals]
        # ques_gs shape: [batch_size, num_rounds, 2]
        # hist_logits shape: [batch_size, num_rounds, num_rounds]
        # ques_gs_prob shape: [batch_size, num_rounds, 2]

        ques_feat = ques
        img_att_ques = self.ATT_MODULE(img, ques_feat)
        return img_att_ques
