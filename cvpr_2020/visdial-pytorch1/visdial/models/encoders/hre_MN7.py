import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import utilities_questioner_predict as utils
from visdial.models.encoders.MN5 import MemN2N
from ban import base_model
from rva.RVA import ATT_MODULE
from ban.fc import FCNet
from rva.classifier import SimpleClassifier
class Encoder(nn.Module):
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
        super(Encoder, self).__init__()
        self.vocabSize = vocabSize
        self.embedSize = embedSize
        self.rnnHiddenSize = rnnHiddenSize
        self.numLayers = numLayers

        # assert self.numLayers > 1, "Less than 2 layers not supported!"
        if useIm:
            self.useIm = useIm if useIm != True else 'early'
        else:
            self.useIm = False
        self.imgEmbedSize = 512
        self.imgFeatureSize = 4096
        self.numRounds = numRounds
        self.dropout = dropout
        self.isAnswerer = isAnswerer
        self.startToken = startToken
        self.endToken = endToken

        # modules
        self.wordEmbed = nn.Embedding(
            self.vocabSize, self.embedSize,
            padding_idx=0)  # changable size of vocabulary padding to vocabSize embedded to embedSize
        # with 0
        # question encoder
        # image fuses early with words
        if self.useIm == 'early':
            quesInputSize = self.embedSize + self.imgEmbedSize
            self.imgNet = nn.Linear(self.imgFeatureSize, self.imgEmbedSize)
            self.imgEmbedDropout = nn.Dropout(0.5)
        elif self.useIm == 'late':
            quesInputSize = self.embedSize  # (300)
            self.imgNet = nn.Linear(self.imgFeatureSize, self.imgEmbedSize)  # (4096,300)
            self.imgEmbedDropout = nn.Dropout(0.5)
        elif self.isAnswerer:
            quesInputSize = self.embedSize
        else:
            quesInputSize = self.embedSize
        if self.isAnswerer:
            self.quesRNN = nn.LSTM(
                quesInputSize,
                self.rnnHiddenSize,
                self.numLayers,
                batch_first=True,
                dropout=self.dropout)
            self.Wq_1 = nn.Linear(self.rnnHiddenSize, self.rnnHiddenSize)
            self.Wh_1 = nn.Linear(self.rnnHiddenSize, self.rnnHiddenSize)
            self.fc1 = nn.Linear(2 * self.rnnHiddenSize, self.rnnHiddenSize)
            self.dialogRNN = nn.LSTM(self.rnnHiddenSize, self.rnnHiddenSize, self.numLayers, batch_first=False,
                                     dropout=self.dropout)
            self.q_net = FCNet([self.rnnHiddenSize, self.rnnHiddenSize])
            self.v_net = FCNet([self.rnnHiddenSize, self.rnnHiddenSize])

        else:
            self.ansRNN = nn.LSTM(
                quesInputSize,
                self.rnnHiddenSize,
                self.numLayers,
                batch_first=True,
                dropout=dropout)
            self.Wq_1 = nn.Linear(self.rnnHiddenSize, self.rnnHiddenSize)
            self.Wh_1 = nn.Linear(self.rnnHiddenSize, self.rnnHiddenSize)
            self.fc1 = nn.Linear(3 * self.rnnHiddenSize, self.rnnHiddenSize)
            self.dialogRNN = nn.LSTM(self.rnnHiddenSize, self.rnnHiddenSize, self.numLayers, batch_first=False,
                                     dropout=self.dropout)
        # history encoder
        self.factRNN = nn.LSTM(
            self.embedSize,  # (300)
            self.rnnHiddenSize,  # (512)
            self.numLayers,  # (2)
            batch_first=True,
            dropout=dropout)

        # define MN
        settings = {
            # "use_cuda": True,
            "num_vocab": self.vocabSize,
            "embedding_dim": self.embedSize,
            "sentence_size": 32,
            "max_hops": 1,
            "dropout": self.dropout
        }
        self.men_n2n = MemN2N(settings)
        self.ban = base_model.build_ban(v_dim=512, num_hid=512, gamma=4).cuda()
        self.ATTMODULE = ATT_MODULE()
        # self.mem_n2n = self.mem_n2n.cuda()

    def reset(self):
        # batchSize is inferred from input
        self.batchSize = 0

        # Input data
        self.image = None
        self.imageEmbed = None

        self.captionTokens = None
        self.captionEmbed = None
        self.captionLens = None

        self.questionTokens = []
        self.questionEmbeds = []
        self.questionLens = []

        self.answerTokens = []
        self.answerEmbeds = []
        self.answerLengths = []

        # Hidden embeddings
        self.factEmbeds = []  # factEmbed(rnn output), states(hidden states, cell states)
        self.questionRNNStates = []
        self.answerRNNStates = []
        self.dialogRNNInputs = []  # rnn_output(hidden states[-1]) (20,512)
        self.dialogHiddens = []  # hidden states (2,20,512)
        self.inptInputs = []

    def _initHidden(self):
        '''Initial dialog rnn state - initialize with zeros'''
        # Dynamic batch size inference
        assert self.batchSize != 0, 'Observe something to infer batch size.'
        someTensor = self.dialogRNN.weight_hh.data
        h = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        c = someTensor.new(self.batchSize, self.dialogRNN.hidden_size).zero_()
        return (Variable(h), Variable(c))

    def observe(self,  # encoder take image/caption/ques/ans as memory
                round,
                image=None,
                caption=None,
                ques=None,
                ans=None,
                captionLens=None,
                quesLens=None,
                ansLens=None):
        if image is not None:
            assert round == -1
            self.image = image
            self.imageEmbed = None
            self.batchSize = len(self.image)
        if caption is not None:
            assert round == -1
            assert captionLens is not None, "Caption lengths required!"
            caption, captionLens = self.processSequence(caption, captionLens)
            self.captionTokens = caption
            self.captionLens = captionLens
            self.batchSize = len(self.captionTokens)
        if ques is not None:
            assert round == len(self.questionEmbeds)  # with ques, round !=-1
            assert quesLens is not None, "Questions lengths required!"
            ques, quesLens = self.processSequence(ques, quesLens)
            self.questionTokens.append(ques)
            self.questionLens.append(quesLens)
        if ans is not None:  # with ans, round !=-1
            assert round == len(self.answerEmbeds)
            assert ansLens is not None, "Answer lengths required!"
            ans, ansLens = self.processSequence(ans, ansLens)
            self.answerTokens.append(ans)
            self.answerLengths.append(ansLens)

    def processSequence(self, seq, seqLen):
        ''' Strip <START> and <END> token from a left-aligned sequence -> only strip <start>'''
        return seq[:, 1:], seqLen - 1

    def embedInputDialog(self):
        # Embed image, occurs once per dialog
        #if self.isAnswerer and self.imageEmbed is None:
        #    self.imageEmbed = F.tanh(self.imgNet(self.imgEmbedDropout(self.image)))
        # Embed caption, occurs once per dialog
        if self.captionEmbed is None:
            self.captionEmbed = self.wordEmbed(self.captionTokens)  # (20,21) => (20,21,300)
        # Embed questions
        while len(self.questionEmbeds) < len(self.questionTokens):  # (20,15)
            idx = len(self.questionEmbeds)
            self.questionEmbeds.append(self.wordEmbed(self.questionTokens[idx]))
        # Embed answers
        while len(self.answerEmbeds) < len(self.answerTokens):
            idx = len(self.answerEmbeds)
            self.answerEmbeds.append(self.wordEmbed(self.answerTokens[idx]))

    def embedFact(self, factIdx):
        # Caption
        if factIdx == 0:
            seq, seqLens = self.captionEmbed, self.captionLens
            factEmbed, states = utils.dynamicRNN(self.factRNN, seq, seqLens, returnStates=True)
            # states: hidden state & cell states(two layers) . factEmbed: rnn output(one layer)
        # QA pairs
        elif factIdx > 0:
            quesTokens, quesLens = \
                self.questionTokens[factIdx - 1], self.questionLens[factIdx - 1]
            ansTokens, ansLens = \
                self.answerTokens[factIdx - 1], self.answerLengths[factIdx - 1]

            qaTokens = utils.concatPaddedSequences(  # concat non-0-token (q,a) and pad with 0 to maxlength
                quesTokens, quesLens, ansTokens, ansLens, padding='right')
            qa = self.wordEmbed(qaTokens)
            qaLens = quesLens + ansLens
            # states: hidden state & cell states(two layers 2*2*20*512) . factEmbed: rnn output(one layer 20*512)
            qaEmbed, states = utils.dynamicRNN(self.factRNN, qa, qaLens, returnStates=True)
            factEmbed = qaEmbed

        factRNNstates = states  # 2[1,20,512]
        self.factEmbeds.append((factEmbed, factRNNstates))


    def embedQuestion(self, qIdx):  # find the longest sentense(chat_processed_data.h5) and pad
        # pack = nn_utils.rnn.pack_padded_sequence(tensor_in, seq_lengths, batch_first=True)
        '''Embed questions'''
        quesIn = self.questionEmbeds[qIdx]
        quesLens = self.questionLens[qIdx]
        if self.useIm == 'early':
            image = self.imageEmbed.unsqueeze(1).repeat(1, quesIn.size(1), 1)
            quesIn = torch.cat([quesIn, image], 2)
        q, qEmbed, states = utils.dynamicRNN1(
            self.quesRNN, quesIn, quesLens)
        quesRNNstates = states  # 2[1,20,512]
        self.questionRNNStates.append((qEmbed, quesRNNstates))
        img_feat_bottom, att = self.ATTMODULE(self.image, qEmbed)
        return img_feat_bottom

    def embedAnswer(self, aIdx):
        '''Embed questions'''
        ansIn = self.answerEmbeds[aIdx]
        ansLens = self.answerLengths[aIdx]
        aEmbed, states = utils.dynamicRNN(
            self.ansRNN, ansIn, ansLens, returnStates=True)
        ansRNNStates = states
        self.answerRNNStates.append((aEmbed, ansRNNStates))

    def concatDialogRNNInput_a(self, histIdx, img_feat_bottom, Att):
        q_repr = self.q_net(self.questionRNNStates[histIdx][0])
        v_repr = self.v_net(img_feat_bottom)
        joint_repr = q_repr * v_repr
        currIns = [Att, joint_repr]  # [512,512,512]
        #if self.useIm == 'late':
        #    currIns.append(self.imageEmbed)
        hist_t = torch.cat(currIns, -1)  # [0][20,512]->[20,512] concat
        self.dialogRNNInputs.append(hist_t)


    def concatDialogRNNInput_q(self, histIdx, Att):
        currIns = [self.factEmbeds[-1][0], Att, self.factEmbeds[0][0]]
        hist_t = torch.cat(currIns, -1)  # [0][20,512]->[20,512] concat
        self.dialogRNNInputs.append(hist_t)

    def embedDialog(self, dialogIdx):
        if self.isAnswerer:
            if dialogIdx == 0:
                hPrev = self.questionRNNStates[-1][1]
            else:
                hPrev = self.dialogHiddens[-1]
            inpt = self.dialogRNNInputs[dialogIdx]
            inpt = F.tanh(self.fc1(F.dropout(inpt, self.dropout, training=True))).unsqueeze(0)  # [1324->512]
            _, hNew = self.dialogRNN(inpt, hPrev)  # [1.20,512], 2[2,20,512]
            self.dialogHiddens.append(hNew)  # 2[1,20,512] hidden states
        else:
            if dialogIdx == 0:
                hPrev = self.factEmbeds[-1][1]
                inpt = self.dialogRNNInputs[dialogIdx]
                inpt = F.tanh((F.dropout(inpt, self.dropout, training=True))).unsqueeze(0)
                self.inptInputs.append(inpt.squeeze(0))
                _, hNew = self.dialogRNN(inpt, hPrev)
                self.dialogHiddens.append(hNew)
            else:
                hPrev = self.dialogHiddens[-1]
                inpt = self.dialogRNNInputs[dialogIdx]
                inpt = F.tanh(self.fc1(F.dropout(inpt, self.dropout, training=True))).unsqueeze(0)
                self.inptInputs.append(inpt.squeeze(0))
                _, hNew = self.dialogRNN(inpt,
                                         hPrev)  # Q-Bot: rnnHiddenSize,rnnHiddenSize A-Bot: 2*rnnHiddenSize, rnnHiddenSize
                self.dialogHiddens.append(hNew)  # [2,20,512] hidden states

    def forward(self):
        '''
        Returns:
            A tuple of tensors (H, C) each of shape (batchSize, rnnHiddenSize)
            to be used as the initial Hidden and Cell states of the Decoder.
            See notes at the end on how (H, C) are computed.
        '''

        # embed all the unembed (Image, Captions, Questions, Answers)
        self.embedInputDialog()

        if self.isAnswerer:
            round = len(self.questionEmbeds) - 1
        else:
            # For Q-Bot, current round is the number of facts present,
            # which is same as the number of answers observed
            round = len(self.answerEmbeds)  # caption & Q-Bot first round =>0

        while len(self.factEmbeds) <= round:  # embed when with caption or with (Q,A) pair
            factIdx = len(self.factEmbeds)
            self.embedFact(factIdx)

        # Embed any un-embedded questions (A-Bot only)
        if self.isAnswerer:
            while len(self.questionRNNStates) <= round:
                qIdx = len(self.questionRNNStates)
                image_feature_bottom = self.embedQuestion(qIdx)  # question Encoder
        else:
            while round >= 1 and len(self.answerRNNStates) < round:
                aIdx = len(self.answerRNNStates)
                self.embedAnswer(aIdx)  # question Encoder

        while len(self.dialogRNNInputs) <= round:
            histIdx = len(self.dialogRNNInputs)
            if self.isAnswerer:
                temp_fact = []
                for i in range(len(self.factEmbeds)):
                    his = self.Wh_1(self.factEmbeds[i][0])
                    temp_fact.append(his)
                ques = self.Wq_1(self.questionRNNStates[histIdx][0])
                fact = torch.stack(temp_fact).permute(1, 0, 2)
                att = self.men_n2n(fact, ques)
                self.concatDialogRNNInput_a(histIdx, image_feature_bottom, att)

            else:
                if histIdx != 0:
                    temp_fact = []
                    for i in range(len(self.factEmbeds) - 1):
                        his = self.Wh_1(self.factEmbeds[i][0])
                        temp_fact.append(his)
                    fact = self.Wq_1(self.factEmbeds[-1][0])
                    att = self.men_n2n(torch.stack(temp_fact).permute(1, 0, 2), fact)
                    self.concatDialogRNNInput_q(histIdx, att)
                else:
                    self.dialogRNNInputs.append(self.factEmbeds[histIdx][0])

        while len(self.dialogHiddens) <= round:
            dialogIdx = len(self.dialogHiddens)
            self.embedDialog(dialogIdx)

        if self.isAnswerer:
            dialogHidden = self.dialogHiddens[-1]
            return dialogHidden  # 2[1,20,512]
        else:
            dialogHidden = self.dialogHiddens[-1]
            imgEmbed = self.inptInputs[-1]
            return dialogHidden, imgEmbed
