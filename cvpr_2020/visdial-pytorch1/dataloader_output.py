import os
import json
import h5py
import numpy as np
import torch
from six import iteritems
from six.moves import range
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset


class VisDialDataset(Dataset):
    def __init__(self, params, subsets):
        '''
            Initialize the dataset with splits given by 'subsets', where
            subsets is taken from ['train', 'val', 'test']

            Notation:
                'dtype' is a split taking values from ['train', 'val', 'test']
                'stype' is a sqeuence type from ['ques', 'ans']
        '''

        # By default, load Q-Bot, A-Bot and dialog options for A-Bot
        self.useQuestion = True
        self.useAnswer = True
        self.useOptions = True
        self.useHistory = True
        self.useIm = True

        # Absorb parameters
        for key, value in iteritems(params):
            setattr(self, key, value)  # if nor exist, create a new attribute in the object
        self.subsets = tuple(subsets)
        self.numRounds = params['numRounds']

        print('\nDataloader loading json file: /hhd/lvxinyu/aqm_plus/resources/data/visdial_0.9'
              '/chat_processed_params.json')
        # with open(self.inputJson, 'r') as fileId:
        with open('/hhd/lvxinyu/aqm_plus/resources/data/visdial_0.9/chat_processed_params.json',
                  'r') as fileId:  # spilt image names, word2ind, ind2word
            info = json.load(fileId)
            # Absorb values
            for key, value in iteritems(info):
                setattr(self, key, value)

        wordCount = len(self.word2ind)
        # Add <START> and <END> to vocabulary
        self.word2ind['<START>'] = wordCount + 1  # 7824
        self.word2ind['<END>'] = wordCount + 2  # 7825
        self.startToken = self.word2ind['<START>']
        self.endToken = self.word2ind['<END>']
        # Padding token is at index 0
        self.vocabSize = wordCount + 3  # 0 ,7824, 7825
        print('Vocab size with <START>, <END>: %d' % self.vocabSize)

        # Construct the reverse map
        self.ind2word = {
            int(ind): word
            for word, ind in iteritems(self.word2ind)
        }

        # Read questions, answers and options
        # print('Dataloader loading h5 file: ' + self.inputQues)
        '''quesFile contains train/val (iosMap and capMap)'''
        quesFile = h5py.File('/hhd/lvxinyu/aqm_plus/resources/data/visdial_0.9/chat_processed_data.h5', 'r')

        if self.useIm:  # if "string"= if True
            # Read images
            # print('Dataloader loading h5 file: ' + self.inputImg)
            imgFile = h5py.File('/hhd/lvxinyu/aqm_plus/resources/data/visdial_0.9/data_img.h5', 'r')

        # Number of data points in each split (train/val/test)
        self.numDataPoints = {}
        self.data = {}

        '''split and save dataset '''
        for dtype in subsets:  # dtype: [train, val, test]
            print("\nProcessing split [%s]..." % dtype)
            if ('ques_%s' % dtype) not in quesFile:
                self.useQuestion = False
            if ('ans_%s' % dtype) not in quesFile:
                self.useAnswer = False
            if ('opt_%s' % dtype) not in quesFile:
                self.useOptions = False

            dataMat_ques = np.array(quesFile['ques_%s' % dtype], dtype='int64')  # ans_train/ques_train (80000,10,20)
            dataMat_ans = np.array(quesFile['ans_%s' % dtype], dtype='int64')
            dataMat_cap = np.array(quesFile['cap_%s' % dtype], dtype='int32')
            whole_dialog = {}
            for i in range(len(dataMat_ques)):
                dialog = {'caption': {}, 0: {}, 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {}}
                caption = []
                for word in dataMat_cap[i]:
                    if word in self.ind2word.keys():
                        caption.append(self.ind2word[word] + ' ')
                dialog['caption'] = ''.join(caption)
                for round in range(len(dataMat_ques[i])):
                    answer = []
                    question = []
                    for word in range(len(dataMat_ques[i][round])):
                        if dataMat_ques[i][round][word] in self.ind2word.keys():
                            question.append(self.ind2word[dataMat_ques[i][round][word]] + ' ')
                        if dataMat_ans[i][round][word] in self.ind2word.keys():
                            answer.append(self.ind2word[dataMat_ans[i][round][word]] + ' ')
                    dialog[round]['question'] = ''.join(question)
                    dialog[round]['answer'] = ''.join(answer)
                whole_dialog[info['unique_img_%s' % dtype][i]] = dialog
                print(dtype, i)
            with open("/hhd/lvxinyu/aqm_plus/resources/data/visdial_0.9/dialog_test.json", "w") as f:
                json.dump(whole_dialog, f)


