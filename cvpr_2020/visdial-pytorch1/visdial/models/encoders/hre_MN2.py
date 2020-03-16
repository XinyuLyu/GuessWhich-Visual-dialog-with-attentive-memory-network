import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import utilities as utils
from visdial.models.encoders.MN3 import MemN2N


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
        assert self.numLayers > 1, "Less than 2 layers not supported!"
        if useIm:
            self.useIm = useIm if useIm != True else 'early'
        else:
            self.useIm = False
        self.imgEmbedSize = imgEmbedSize
        self.imgFeatureSize = imgFeatureSize
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
            dialogInputSize = 2 * self.rnnHiddenSize
            self.imgNet = nn.Linear(self.imgFeatureSize, self.imgEmbedSize)
            self.imgEmbedDropout = nn.Dropout(0.5)
        elif self.useIm == 'late':
            quesInputSize = self.embedSize  # (300)
            dialogInputSize = 2*self.rnnHiddenSize + self.imgEmbedSize  # why? 2 history encoder layer
            self.imgNet = nn.Linear(self.imgFeatureSize, self.imgEmbedSize)  # (4096,300)
            self.imgEmbedDropout = nn.Dropout(0.5)
        elif self.isAnswerer:
            quesInputSize = self.embedSize
            dialogInputSize = 2 * self.rnnHiddenSize
        else:
            quesInputSize = self.embedSize
            dialogInputSize = 3 * self.rnnHiddenSize
        if self.isAnswerer:
            self.quesRNN = nn.LSTM(
                quesInputSize,
                self.rnnHiddenSize,
                self.numLayers,
                batch_first=True,
                dropout=0)
        else:
            self.ansRNN = nn.LSTM(
                quesInputSize,
                self.rnnHiddenSize,
                self.numLayers,
                batch_first=True,
                dropout=0)
            self.Dense = nn.Linear(2 * rnnHiddenSize, rnnHiddenSize)
        # history encoder
        self.factRNN = nn.LSTM(
            self.embedSize,  # (300)
            self.rnnHiddenSize,  # (512)
            self.numLayers,  # (2)
            batch_first=True,
            dropout=0)

        # dialog rnn
        self.dialogRNN = nn.LSTMCell(dialogInputSize, self.rnnHiddenSize)

        # define MN
        settings = {
            # "use_cuda": True,
            "num_vocab": self.vocabSize,
            "embedding_dim": self.embedSize,
            "sentence_size": 32,
            "max_hops": 1
        }
        self.men_n2n = MemN2N(settings)
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
        '''
        Store dialog input to internal model storage

        Note that all input sequences are assumed to be left-aligned (i.e.
        right-padded). Internally this alignment is changed to right-align
        for ease in computing final time step hidden states of each RNN
        '''
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
        '''
        Lazy embedding of input:
            Calling observe does not process (embed) any inputs. Since
            self.forward requires embedded inputs, this function lazily
            embeds them so that they are not re-computed upon multiple
            calls to forward in the same round of dialog.
        '''
        # Embed image, occurs once per dialog
        if self.isAnswerer and self.imageEmbed is None:
            self.imageEmbed = self.imgNet(self.imgEmbedDropout(self.image))  # fc (4096,300)
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
        '''Embed facts i.e. caption and round 0 or question-answer pair otherwise'''
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

        factRNNstates = states
        self.factEmbeds.append((factEmbed, factRNNstates))

    def embedQuestion(self, qIdx):
        '''Embed questions'''
        quesIn = self.questionEmbeds[qIdx]
        quesLens = self.questionLens[qIdx]
        if self.useIm == 'early':
            image = self.imageEmbed.unsqueeze(1).repeat(1, quesIn.size(1), 1)
            quesIn = torch.cat([quesIn, image], 2)
        qEmbed, states = utils.dynamicRNN(
            self.quesRNN, quesIn, quesLens, returnStates=True)
        quesRNNstates = states
        self.questionRNNStates.append((qEmbed, quesRNNstates))

    def embedAnswer(self, aIdx):
        '''Embed questions'''
        ansIn = self.answerEmbeds[aIdx]
        ansLens = self.answerLengths[aIdx]
        aEmbed, states = utils.dynamicRNN(
            self.ansRNN, ansIn, ansLens, returnStates=True)
        ansRNNStates = states
        self.answerRNNStates.append((aEmbed, ansRNNStates))

    def concatDialogRNNInput(self, histIdx, Att):
        currIns = [Att,self.questionRNNStates[histIdx][0]]
        if self.useIm == 'late':
            currIns.append(self.imageEmbed)
        hist_t = torch.cat(currIns, -1)  # [0][20,512]->[20,512] concat
        self.dialogRNNInputs.append(hist_t)

    def concatDialogRNNInput_q(self, histIdx, Att):
        currIns = [self.factEmbeds[-1][0],Att,self.factEmbeds[0][0]]
        hist_t = torch.cat(currIns, -1)  # [0][20,512]->[20,512] concat
        self.dialogRNNInputs.append(hist_t)

    def embedDialog(self, dialogIdx):
        if dialogIdx == 0:
            hPrev = self._initHidden()
        else:
            hPrev = self.dialogHiddens[-1]
        inpt = self.dialogRNNInputs[dialogIdx]
        hNew = self.dialogRNN(inpt, hPrev)  # Q-Bot: rnnHiddenSize,rnnHiddenSize A-Bot: 2*rnnHiddenSize, rnnHiddenSize
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
            # For A-Bot, current round is the number of facts present,
            # which is number of questions observed - 1 (as opposed
            # to len(self.answerEmbeds), which may be inaccurate as
            round = len(self.questionEmbeds) - 1
        else:
            # For Q-Bot, current round is the number of facts present,
            # which is same as the number of answers observed
            round = len(self.answerEmbeds)  # caption & Q-Bot first round =>0

        # Lazy computation of internal hidden embeddings (hence the while loops)

        # embed questions/answers/captions  ->Fact Embedding (with concat)
        while len(self.factEmbeds) <= round:  # embed when with caption or with (Q,A) pair
            factIdx = len(self.factEmbeds)
            self.embedFact(factIdx)

        # Embed any un-embedded questions (A-Bot only)
        if self.isAnswerer:
            while len(self.questionRNNStates) <= round:
                qIdx = len(self.questionRNNStates)
                self.embedQuestion(qIdx)  # question Encoder
        else:
            while round >= 1 and len(self.answerRNNStates) < round:
                aIdx = len(self.answerRNNStates)
                self.embedAnswer(aIdx)  # question Encoder

        # gather input(factembed.rnn_outputs[20,512]) to dialogRNN
        while len(self.dialogRNNInputs) <= round:
            histIdx = len(self.dialogRNNInputs)
            if self.isAnswerer:
                temp_fact = []
                for i in range(len(self.factEmbeds)):
                    temp_fact.append(self.factEmbeds[i][0])
                att = self.men_n2n(torch.stack(temp_fact).permute(1,0,2), self.questionRNNStates[histIdx][0])# histIdx or -1 ???
                self.concatDialogRNNInput(histIdx, att)
            else:
                temp_fact = []
                for i in range(len(self.factEmbeds)-1):
                    temp_fact.append(self.factEmbeds[i][0])
                att = self.men_n2n(torch.stack(temp_fact).permute(1,0,2),self.factEmbeds[-1][0])
                self.concatDialogRNNInput_q(histIdx,att)

        # Forward dialogRNN -> History Encoder forward
        while len(self.dialogHiddens) <= round:
            dialogIdx = len(self.dialogHiddens)
            self.embedDialog(dialogIdx)

        # Latest dialogRNN hidden state
        dialogHidden = self.dialogHiddens[-1][0]  # (20,512)

        '''
        Return hidden (H_link) and cell (C_link) states as per the following rule:
        (Currently this is defined only for numLayers == 2)
        If A-Bot:
          C_link == Question encoding RNN cell state (quesRNN) (Layer 0 & Layer 1)
          H_link ==
              Layer 0 : Question encoding RNN hidden state (quesRNN)
              Layer 1 : DialogRNN hidden state (dialogRNN) (rnn_output)

        If Q-Bot:
            C_link == Fact encoding RNN cell state (factRNN) (Layer 0 & Layer 1)
            H_link ==
                Layer 0 : Fact encoding RNN hidden state (factRNN)
                Layer 1 : DialogRNN hidden state (dialogRNN) (rnn_output) 
        '''
        if self.isAnswerer:
            quesRNNstates = self.questionRNNStates[-1][1]  # Latest quesRNN states
            C_link = quesRNNstates[1]
            H_link = quesRNNstates[0][:-1]
            H_link = torch.cat([H_link, dialogHidden.unsqueeze(0)], 0)
        else:
            factRNNstates = self.factEmbeds[-1][1]  # [0][0[(20,512)],1[(2,20,512),(2,20,512)]]
            C_link = factRNNstates[1]  # (2,20,512)
            H_link = factRNNstates[0][:-1]  # (1,20,512)
            H_link = torch.cat([H_link, dialogHidden.unsqueeze(0)],
                               0)  # (1,20,512)fact+(1,20,512)dialoghidden=(2,20,512)

        return H_link, C_link
