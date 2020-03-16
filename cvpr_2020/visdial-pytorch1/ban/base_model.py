"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is written by Jin-Hwa Kim.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from ban.attention import BiAttention

from ban.fc import FCNet
from ban.bc import BCNet


class BanModel(nn.Module):
    def __init__(self, v_att, b_net, q_prj, c_prj, glimpse):
        super(BanModel, self).__init__()
        #self.dataset = dataset
        #self.op = op
        self.glimpse = glimpse
        #self.w_emb = w_emb
        #self.q_emb = q_emb
        self.v_att = v_att
        self.b_net = nn.ModuleList(b_net)
        self.q_prj = nn.ModuleList(q_prj)
        self.c_prj = nn.ModuleList(c_prj)
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, v, q_emb):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        #w_emb = self.w_emb(q) # word embedding

        #q_emb = self.q_emb.forward_all(w_emb) # [batch, q_len, q_dim] all previous hidden states
        #boxes = b[:,:,:4].transpose(1,2)

        b_emb = [0] * self.glimpse #
        att, logits = self.v_att.forward_all(v, q_emb) # b x g x v x q calulate attention map

        for g in range(self.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:,g,:,:]) # b x l x h resnet learning
            #atten, _ = logits[:,g,:,:].max(2)
            #embed = self.counter(boxes, atten)
            temp = self.q_prj[g](b_emb[g].unsqueeze(1))#.data #[20,1,512] float tensor [20,14,512]
            q_emb += temp
            #q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        logits = q_emb.sum(1)
        #logits = self.classifier(q_emb.sum(1))

        return logits, att


def build_ban(v_dim, num_hid, gamma):
    #w_emb = WordEmbedding(dataset.dictionary.ntoken, 300, .0, op)
    #q_emb = QuestionEmbedding(300 if 'c' not in op else 600, num_hid, 1, False, .0)
    #v_att = BiAttention(dataset.v_dim, num_hid, num_hid, gamma)
    v_att = BiAttention(v_dim, num_hid, num_hid, gamma)
    b_net = []
    q_prj = []
    c_prj = []
    objects = 10  # minimum number of boxes
    for i in range(gamma):
        b_net.append(BCNet(v_dim, num_hid, num_hid, None, k=1))
        q_prj.append(FCNet([num_hid, num_hid], '', .2))
        c_prj.append(FCNet([objects + 1, num_hid], 'ReLU', .0))
    return BanModel(v_att, b_net, q_prj, c_prj, gamma)

if __name__=='__main__':
    '''
    net = BCNet(1024,1024,1024,1024).cuda()
    x = torch.Tensor(512,36,1024).cuda()
    y = torch.Tensor(512,14,1024).cuda()
    out = net.forward(x,y)
    print(net)
    print(out)'''
    v = torch.Tensor(20, 36, 512).cuda()
    q = torch.Tensor(20, 14, 512).cuda() #[batch, hidden_states/sequence_length, h_out]
    fc1 = torch.nn.Linear(300,512)

    d=Variable(torch.rand(20, 36, 300))
    d = fc1(d)
    model = build_ban(v_dim=512, num_hid=512, gamma=4)
    model.cuda()
    logits, att = model(v, q)  # [20, 512]
    print()