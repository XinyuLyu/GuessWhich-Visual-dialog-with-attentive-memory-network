import sys
import json
import gc
import h5py
import numpy as np
from timeit import default_timer as timer

import torch
from torch.autograd import Variable

import options
import visdial.metrics as metrics
from utils import utilities as utils
from dataloader import VisDialDataset
from torch.utils.data import DataLoader

from sklearn.metrics.pairwise import pairwise_distances

from six.moves import range


def dialogDump(params,
               dataset,
               split,
               aBot,
               qBot=None,
               beamSize=1,
               expLowerLimit=None,
               expUpperLimit=None,
               savePath="dialog_results.json"):
    '''
        Generates dialog and saves it to a json for later visualization.
        If only A-Bot is given, A-Bot answers are generated given GT image,
        caption and questions. If both agents are given, dialog is generated
        by both agents conversing (A-Bot is shown the GT image and both
        agents have access to a caption generated by a pre-trained captioning
        model).

        Arguments:
            params  : Parameter dict for all options
            dataset : VisDialDataset instance
            split   : Dataset split, can be 'val' or 'test'
            aBot    : A-Bot
            qBot    : Q-Bot (Optional)

            beamSize : Beam search width for generating utterrances
            savePath : File path for saving dialog json file
    '''
    assert aBot is not None or (qBot is not None and aBot is not None),\
                            "Must provide either an A-Bot alone or both \
                            Q-Bot and A-Bot when generating dialog"
    old_split = dataset.split
    batchSize = dataset.batchSize
    # hack here
    numRounds = dataset.numRounds
    dataset.split = split
    ind2word = dataset.ind2word
    to_str_gt = lambda w: str(" ".join([ind2word[x] for x in filter(lambda x:\
                    x>0,w.data.cpu().numpy())])) #.encode('utf-8','ignore')
    to_str_pred = lambda w, l: str(" ".join([ind2word[x] for x in list( filter(
        lambda x:x>0,w.data.cpu().numpy()))][:l.data.cpu()[0]])) #.encode('utf-8','ignore')

    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn)

    text = {'data': []}
    if '%s_img_fnames' % split not in dataset.data.keys():
        print("[Error] Need coco directory and info as input " \
               "to -cocoDir and -cocoInfo arguments for locating "\
               "coco image test_results.")
        print("Exiting dialogDump without saving test_results.")
        return None

    getImgFileName = lambda x: dataset.data['%s_img_fnames' % split][x]
    getImgId = lambda x: int(getImgFileName(x)[:-4][-12:])

    for idx, batch in enumerate(dataloader):
        if expLowerLimit is not None:
            if idx < expLowerLimit // batchSize: continue
            if idx > expUpperLimit // batchSize: break
        else:
            if idx > 40:
                break
        imgIds = [getImgId(x) for x in batch['index']]
        dialog = [{'dialog': [], 'image_id': imgId} for imgId in imgIds]

        if dataset.useGPU:
            batch = {key: v.cuda() if hasattr(v, 'cuda')\
                else v for key, v in batch.items()}

        image = Variable(batch['img_feat'], volatile=True)
        caption = Variable(batch['cap'], volatile=True)
        captionLens = Variable(batch['cap_len'], volatile=True)
        if qBot is None:  # A-Bot alone needs ground truth dialog
            gtQuestions = Variable(batch['ques'], volatile=True)
            gtQuesLens = Variable(batch['ques_len'], volatile=True)
            gtAnswers = Variable(batch['ans'], volatile=True)
            gtAnsLens = Variable(batch['ans_len'], volatile=True)

        if aBot:
            aBot.eval(), aBot.reset()
            aBot.observe(
                -1, image=image, caption=caption, captionLens=captionLens)
        if qBot:
            qBot.eval(), qBot.reset()
            qBot.observe(-1, caption=caption, captionLens=captionLens)
        questions = []

        for j in range(batchSize):
            caption_str = to_str_gt(caption[j])[8:-6]
            dialog[j]['caption'] = caption_str

        for round in range(numRounds):
            if aBot is not None and qBot is None:
                aBot.observe(
                    round,
                    ques=gtQuestions[:, round],
                    quesLens=gtQuesLens[:, round])
                aBot.observe(
                    round,
                    ans=gtAnswers[:, round],
                    ansLens=gtAnsLens[:, round])
                _ = aBot.forward()
                answers, ansLens = aBot.forwardDecode(
                    inference='greedy', beamSize=beamSize)

            elif aBot is not None and qBot is not None:
                questions, quesLens = qBot.forwardDecode(
                    beamSize=beamSize, inference='greedy')
                qBot.observe(round, ques=questions, quesLens=quesLens)
                aBot.observe(round, ques=questions, quesLens=quesLens)
                answers, ansLens = aBot.forwardDecode(
                    beamSize=beamSize, inference='greedy')
                aBot.observe(round, ans=answers, ansLens=ansLens)
                qBot.observe(round, ans=answers, ansLens=ansLens)

            for j in range(batchSize):
                question_str = to_str_pred(questions[j], quesLens[j]) \
                    if qBot is not None else to_str_gt(gtQuestions[j])
                answer_str = to_str_pred(answers[j], ansLens[j])

                dialog[j]['dialog'].append({
                    "answer": answer_str[8:],
                    "question": question_str[8:] + " "
                })  # "8:" for indexing out initial <START>
        text['data'].extend(dialog)

    text['opts'] = {
        'qbot': params['qstartFrom'],
        'abot': params['startFrom'],
        'backend': 'cudnn',
        'beamLen': 20,
        'beamSize': beamSize,
        'decoder': params['decoder'],
        'encoder': params['encoder'],
        'gpuid': 0,
        'imgNorm': params['imgNorm'],
        'inputImg': params['inputImg'],
        'inputJson': params['inputJson'],
        'inputQues': params['inputQues'],
        'loadPath': 'checkpoints/',
        'maxThreads': 1,
        'resultPath': 'dialog_output/results',
        'sampleWords': 0,
        'temperature': 1,
        'useHistory': True,
        'useIm': True,
        'AQM': 0,
    }
    with open(savePath, "w") as fp:
        print("Writing dialog text data to file: {}".format(savePath))
        json.dump(text, fp)
    print("Done!")

    dataset.split = old_split
    return