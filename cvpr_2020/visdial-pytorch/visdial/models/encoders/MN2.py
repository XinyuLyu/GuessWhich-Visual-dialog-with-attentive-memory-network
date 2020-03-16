import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


def position_encoding(sentence_size, embedding_dim):
    encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i - 1, j - 1] = (i - (embedding_dim + 1) / 2) * (j - (sentence_size + 1) / 2)
    encoding = 1 + 4 * encoding / embedding_dim / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    encoding[:, -1] = 1.0
    return np.transpose(encoding)


class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class MemN2N(nn.Module):
    def __init__(self, settings):
        super(MemN2N, self).__init__()

        #use_cuda = True
        num_vocab = settings["num_vocab"]
        embedding_dim = 512 #settings["embedding_dim"]
        sentence_size = settings["sentence_size"]
        self.max_hops = settings["max_hops"]

        for hop in range(self.max_hops + 1):
            C = nn.Embedding(num_vocab, embedding_dim, padding_idx=0)
            C.weight.data.normal_(0, 0.1)  # init
            self.add_module("C_{}".format(hop), C)
        self.C = AttrProxy(self, "C_")
        self.fc= nn.Sequential(nn.Linear(512, 512), nn.BatchNorm1d(512),nn.ReLU(True))
        self.softmax = nn.Softmax()
        #self.encoding = Variable(torch.FloatTensor(position_encoding(sentence_size, embedding_dim)), requires_grad=False)#(32,300)

        #if use_cuda:
        #    self.encoding = self.encoding.cuda()

    def forward(self, story, query):# story(20,10,512), query(20,512)
        #story_size = story.size()
        u = list()
        #embedding B
        #query_embed = self.C[0](query) #(32,8)->(32,8,20)
        #encoding = self.encoding.unsqueeze(0).expand_as(query) #(8,20)->(32,8,20) / (32,512)->(20,32,512)
        u.append(query) #(32,20) / (20,300)

        for hop in range(self.max_hops):
            # embedding A
            #embed_A = self.C[hop](story.view(story.size(0), -1)) #story (32,50,8)->(32,400,20)
            #embed_A = embed_A.view(story_size + (embed_A.size(-1),)) #story (32,400,20)->(32,50,8,20)
            #encoding = self.encoding.unsqueeze(0).unsqueeze(1).expand_as(story)#encoding (32,8,20)-> (32,50,8,20)
            #m_A = torch.sum(story * encoding, 2)# (32,50,8,20)-> (32,50,20)

            # attention
            u_temp = u[-1].unsqueeze(1).expand_as(story)#(32,20)->(32,50,20)
            prob = self.softmax(torch.sum(story * u_temp, 2)) #(32,50)

            # embedding C
            #embed_C = self.C[hop + 1](story.view(story.size(0), -1))#(32,50,8)->(32,400,20)
            #embed_C = embed_C.view(story_size + (embed_C.size(-1),))#(32,50,8,20)
            #m_C = torch.sum(embed_C * encoding, 2) #(32,50,20)

            #calculate output
            prob = prob.unsqueeze(2).expand_as(story) #(32,50,20)
            o_k = torch.sum(story * prob, 1)#(32,20)

            # sum
            u_k = u[-1] + self.fc(o_k)
            u.append(u_k)
        return u[-1]
        #a_hat = u[-1] @ self.C[self.max_hops].weight.transpose(0, 1) #(32,85)
        #return a_hat, self.softmax(a_hat)

