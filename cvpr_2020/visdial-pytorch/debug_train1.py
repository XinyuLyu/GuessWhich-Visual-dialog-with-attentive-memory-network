import os
import gc
import random
import pprint
from six.moves import range
from markdown2 import markdown
from time import gmtime, strftime
from timeit import default_timer as timer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import debug_options
from dataloader_vgg_mean_global import VisDialDataset
from torch.utils.data import DataLoader
from eval_utils.rank_answerer1 import rankABot
from eval_utils.rank_questioner import rankQBot
from utils import utilities_questioner_predict as utils
from utils.visualize import VisdomVisualize
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

# ---------------------------------------------------------------------------
# command options
# ---------------------------------------------------------------------------
params = debug_options.readCommandLine()
print("params:", params)
# Seed rng for reproducibility
random.seed(params['randomSeed'])
torch.manual_seed(params['randomSeed'])
if params['useGPU']:
    torch.cuda.manual_seed_all(params['randomSeed'])

# ---------------------------------------------------------------------------
# dataset
# ---------------------------------------------------------------------------
splits = ['train', 'val', 'test']
dataset = VisDialDataset(params, splits)
# extra Params to transfer from dataset
transfer = ['vocabSize', 'numOptions', 'numRounds']
for key in transfer:
    if hasattr(dataset, key):
        params[key] = getattr(dataset, key)

# Create save path and checkpoints folder
# os.makedirs('/hhd/lvxinyu/visdial-pytorch/checkpoints/', exist_ok=True)
os.mkdir(params['savePath'])

# ---------------------------------------------------------------------------
# Loading Modules
# ---------------------------------------------------------------------------
parameters = []  # save params from abot/qbot
aBot = None
qBot = None
# Loading A-Bot
if params['trainMode'] in ['sl-abot', 'rl-full-QAf']:
    aBot, loadedParams, optim_state = utils.loadModel(params, 'abot')
    for key in loadedParams:
        params[key] = loadedParams[key]
    parameters.extend(aBot.parameters())
# Loading Q-Bot
if params['trainMode'] in ['sl-qbot', 'rl-full-QAf']:
    qBot, loadedParams, optim_state = utils.loadModel(params, 'qbot')
    for key in loadedParams:
        params[key] = loadedParams[key]

    if params['trainMode'] == 'rl-full-QAf' and params['freezeQFeatNet']:
        qBot.freezeFeatNet()
    # Filtering parameters whose requires_grad=True
    parameters.extend(filter(lambda p: p.requires_grad, qBot.parameters()))
    # parameters.extend(qBot.parameters())

# ---------------------------------------------------------------------------
# dataloader (pytorch)
# ---------------------------------------------------------------------------
dataset.split = 'train'
dataloader = DataLoader(
    dataset,
    batch_size=params['batchSize'],
    shuffle=False,
    num_workers=params['numWorkers'],
    drop_last=True,
    collate_fn=dataset.collate_fn,
    pin_memory=False)

# ---------------------------------------------------------------------------
# plotting (visdom)
# ---------------------------------------------------------------------------
viz = VisdomVisualize(
    enable=bool(params['enableVisdom']),
    env_name=params['visdomEnv'],
    server=params['visdomServer'],
    port=params['visdomServerPort'])
pprint.pprint(params)
viz.addText(pprint.pformat(params, indent=4))

# ---------------------------------------------------------------------------
# optimizer, loss, numIterPerEpoch, rlRound
# ---------------------------------------------------------------------------
if params['continue']:
    # Continuing from a loaded checkpoint restores the following
    startIterID = params['ckpt_iterid'] + 1  # Iteration ID
    lRate = params['ckpt_lRate']  # Learning rate
    print("Continuing training from iterId[%d]" % startIterID)
else:
    # Beginning training normally, without any checkpoint
    lRate = params['learningRate']
    startIterID = 0
optimizer = optim.Adam(parameters, lr=lRate)
if params['continue']:  # Restoring optimizer state
    print("Restoring optimizer state dict from checkpoint")
    optimizer.load_state_dict(optim_state)
runningLoss = None
mse_criterion = nn.MSELoss(reduce=False)
#mse_criterion = nn.CosineSimilarity(dim=1, eps=1e-8)
numIterPerEpoch = dataset.numDataPoints['train'] // params['batchSize']
print('\n%d iter per epoch.' % numIterPerEpoch)
# RL round
if params['useCurriculum']:
    if params['continue']:
        rlRound = max(0, 9 - (startIterID // numIterPerEpoch))
    else:
        rlRound = params['numRounds'] - 1
else:
    rlRound = 0


# ---------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------

def batch_iter(dataloader):
    for epochId in range(params['numEpochs']):
        for idx, batch in enumerate(dataloader):
            yield epochId, idx, batch


start_t = timer()
for epochId, idx, batch in batch_iter(dataloader):  # 65, 2536, 20 batches of data, each iter different batch
    iterId = startIterID + idx + (epochId * numIterPerEpoch)  # consider continue
    epoch = iterId // numIterPerEpoch
    gc.collect()

    # Moving current batch to GPU, if available
    if dataset.useGPU:
        batch = {key: v.cuda() if hasattr(v, 'cuda') \
            else v for key, v in batch.items()}

    image = Variable(batch['img_feat_mean'], requires_grad=False)  # (20,4096)
    image_global = Variable(batch['img_feat_global'], requires_grad=False)
    caption = Variable(batch['cap'], requires_grad=False)  # (20,22)
    captionLens = Variable(batch['cap_len'], requires_grad=False)  # (20)
    gtQuestions = Variable(batch['ques'], requires_grad=False)  # (20,10,16)
    gtQuesLens = Variable(batch['ques_len'], requires_grad=False)  # (20,10)
    gtAnswers = Variable(batch['ans'], requires_grad=False)  # (20,10,18)
    gtAnsLens = Variable(batch['ans_len'], requires_grad=False)  # (20,10)
    options = Variable(batch['opt'], requires_grad=False)  # (20,10,100,22)
    optionLens = Variable(batch['opt_len'], requires_grad=False)  # (20,10,100)
    gtAnsId = Variable(batch['ans_id'], requires_grad=False)  # (20,10)

    # Initializing optimizer and losses
    optimizer.zero_grad()
    loss = 0
    qBotLoss = 0
    aBotLoss = 0
    rlLoss = 0
    featLoss = 0
    cos_similarity_loss = 0
    huber_loss = 0
    qBotRLLoss = 0
    aBotRLLoss = 0
    predFeatures = None
    initialGuess = None
    numRounds = params['numRounds']
    # numRounds = 1 # Override for debugging lesser rounds of dialog

    # Setting training modes for both bots and observing captions, images where needed
    if aBot:
        aBot.train(), aBot.reset()
        aBot.observe(-1, image=[image,image_global], caption=caption, captionLens=captionLens)
    # 8.12
    if qBot:
        qBot.train(), qBot.reset()
        qBot.observe(-1, caption=caption, captionLens=captionLens)

    # Q-Bot image feature regression ('guessing') only occurs if Q-Bot is present
    if params['trainMode'] in ['sl-qbot', 'rl-full-QAf']:
        initialGuess = qBot.predictImage()
        prevFeatDist = mse_criterion(initialGuess, image)#[20, 4096]
        featLoss += torch.mean(prevFeatDist)  # overall mean (per round) [1]
        prevFeatDist = torch.mean(prevFeatDist, 1)  # mean for each line [20]

    # round 1: gtAnswers[:,0], round 2: gtAnswers[:,1]...
    for round in range(numRounds):
        '''
        Loop over rounds of dialog. Currently three modes of training are
        supported:

            sl-abot :
                Supervised pre-training of A-Bot model using cross
                entropy loss with ground truth answers

            sl-qbot :
                Supervised pre-training of Q-Bot model using cross
                entropy loss with ground truth questions for the
                dialog model and mean squared error loss for image
                feature regression (i.e. image prediction)

            rl-full-QAf :
                RL-finetuning of A-Bot and Q-Bot in a cooperative
                setting where the common reward is the difference
                in mean squared error between the current and
                previous round of Q-Bot's image prediction.

                Annealing: In order to ease in the RL objective,
                fine-tuning starts with first N-1 rounds of SL
                objective and last round of RL objective - the
                number of RL rounds are increased by 1 after
                every epoch until only RL objective is used for
                all rounds of dialog.

        '''
        forwardABot = ((params['trainMode'] == 'sl-abot') or (params['trainMode'] == 'rl-full-QAf' and round < rlRound))
        forwardQBot = ((params['trainMode'] == 'sl-qbot') or (params['trainMode'] == 'rl-full-QAf' and round < rlRound))
        forwardFeatNet = (forwardQBot or params['trainMode'] == 'rl-full-QAf')

        if forwardABot:
            aBot.observe(round, ques=gtQuestions[:, round], quesLens=gtQuesLens[:, round])
            aBot.observe(round, ans=gtAnswers[:, round], ansLens=gtAnsLens[:, round])
            ansLogProbs = aBot.forward()  # (20,18,7826) round 1: (20,18) (answers at round 1)
            aBotLoss += utils.maskedNll(ansLogProbs, gtAnswers[:, round].contiguous())

        if forwardQBot:
            qBot.observe(round, ques=gtQuestions[:, round], quesLens=gtQuesLens[:, round])
            quesLogProbs = qBot.forward()  # (20,16,7286)
            qBotLoss += utils.maskedNll(quesLogProbs, gtQuestions[:, round].contiguous())
            qBot.observe(round, ans=gtAnswers[:, round], ansLens=gtAnsLens[:, round])  # for next step training

        # In order to stay true to the original implementation, the feature
        # regression network makes predictions before dialog begins and for
        # the first 9 rounds of dialog. This can be set to 10 if needed.
        MAX_FEAT_ROUNDS = 9

        # round[0:8] make prediction
        if forwardFeatNet and round < MAX_FEAT_ROUNDS:
            predFeatures = qBot.predictImage()
            featDist = mse_criterion(predFeatures, image)
            featDist = torch.mean(featDist)
            featLoss += featDist

        # Diversity Penalty
        if params["useCosSimilarityLoss"] or params["useHuberLoss"]:
            if params['trainMode'] == 'sl-qbot' or params['trainMode'] == 'rl-full-QAf':
                cur_dialog_hidden = qBot.encoder.dialogHiddens[-1][0][0]
            elif params['trainMode'] == 'sl-abot':
                cur_dialog_hidden = aBot.encoder.dialogHiddens[-1][0][0]

            if params["useCosSimilarityLoss"]:
                if round > 0:
                    cos_similarity_loss += utils.cosinePenalty(cur_dialog_hidden, past_dialog_hidden)

            if params["useHuberLoss"]:
                if round > 0:
                    huber_loss += utils.huberPenalty(cur_dialog_hidden, past_dialog_hidden, threshold=0.1)

            if round == 0:

                if params['trainMode'] == 'sl-qbot' or params['trainMode'] == 'rl-full-QAf':
                    past_dialog_hidden = qBot.encoder.dialogHiddens[-1][0][0]
                elif params['trainMode'] == 'sl-abot':
                    past_dialog_hidden = aBot.encoder.dialogHiddens[-1][0][0]

            else:
                past_dialog_hidden = cur_dialog_hidden
        # round>= rlRound finetune in RL
        if params['trainMode'] == 'rl-full-QAf' and round >= rlRound:
            # Run one round of conversation
            questions, quesLens = qBot.forwardDecode(inference='sample')
            qBot.observe(round, ques=questions, quesLens=quesLens)
            aBot.observe(round, ques=questions, quesLens=quesLens)
            answers, ansLens = aBot.forwardDecode(inference='sample')
            aBot.observe(round, ans=answers, ansLens=ansLens)
            qBot.observe(round, ans=answers, ansLens=ansLens)

            # Q-Bot makes a guess at the end of each round
            predFeatures = qBot.predictImage()

            # Computing reward based on Q-Bot's predicted image
            featDist = mse_criterion(predFeatures, image)
            featDist = torch.mean(featDist, 1)
            # predict_dis(before round 10) - predict_dis(after round 10)
            reward = prevFeatDist.detach() - featDist
            prevFeatDist = featDist

            qBotRLLoss = qBot.reinforce(reward)
            if params['rlAbotReward']:
                aBotRLLoss = aBot.reinforce(reward)
            rlLoss += torch.mean(aBotRLLoss)
            rlLoss += torch.mean(qBotRLLoss)

    # Loss coefficients
    rlCoeff = params['RLLossCoeff']
    rlLoss = rlLoss * rlCoeff
    featLoss = featLoss * params['featLossCoeff']  # 1000 mse loss
    # Averaging over rounds
    qBotLoss = (params['CELossCoeff'] * qBotLoss) / numRounds  # 200
    aBotLoss = (params['CELossCoeff'] * aBotLoss) / numRounds  # 200,10
    featLoss = featLoss / numRounds  # / (numRounds+1)
    rlLoss = rlLoss / numRounds
    cos_similarity_loss = (params['CosSimilarityLossCoeff'] * cos_similarity_loss) / numRounds
    huber_loss = -(params["HuberLossCoeff"] * huber_loss)/numRounds

    loss = qBotLoss + aBotLoss + rlLoss + featLoss + cos_similarity_loss + huber_loss
    loss.backward()
    optimizer.step()

    # Tracking a running average of loss
    if runningLoss is None:
        runningLoss = loss.data[0]  # total loss
    else:
        runningLoss = 0.95 * runningLoss + 0.05 * loss.data[0]  # 0.95*last time + 0.05* this time

    # Decay learning rate after every iteration
    if lRate > params['minLRate']:
        for gId, group in enumerate(optimizer.param_groups):
            optimizer.param_groups[gId]['lr'] *= params['lrDecayRate']
        lRate *= params['lrDecayRate']

    # RL Annealing: Every epoch after the first, decrease rlRound
    if iterId % numIterPerEpoch == 0 and iterId > 0:
        if params['trainMode'] == 'rl-full-QAf':
            rlRound = max(0, rlRound - 1)  # RL round increase by 1 per Epoch
            print('Using rl starting at round {}'.format(rlRound))

    # Print every now and then
    if iterId % 10 == 0:
        end_t = timer()  # Keeping track of iteration(s) time
        curEpoch = float(iterId) / numIterPerEpoch
        timeStamp = strftime('%a %d %b %y %X', gmtime())
        printFormat = '[Ep: %.2f][Iter: %d]][Loss: %.3g]'
        printInfo = [
             curEpoch, iterId, loss.data[0]
        ]
        start_t = end_t
        if isinstance(aBotLoss, Variable):
            printFormat += '[aBotLoss(train CE): %.3g]'
            printInfo.append(aBotLoss.data[0])
        if isinstance(qBotLoss, Variable):
            printFormat += '[qBotLoss(train CE): %.3g]'
            printInfo.append(qBotLoss.data[0])
        if isinstance(rlLoss, Variable):
            printFormat += '[rlLoss(train): %.3g]'
            printInfo.append(rlLoss.data[0])
        if isinstance(featLoss, Variable):
            printFormat += '[featLoss(train FeatureRegressionLoss): %.3g]'
            printInfo.append(featLoss.data[0])
        printFormat += '[Runningloss): %.3g]'
        printInfo.append(runningLoss)
        print(printFormat % tuple(printInfo))

    # ---------------------------------------------------------------------------
    # Evaluate every epoch
    # ---------------------------------------------------------------------------
    if iterId % (numIterPerEpoch // 1) == 0:
        # Set eval mode
        if aBot:
            aBot.eval()
        if qBot:
            qBot.eval()

        print('Performing validation...')
        if aBot and 'ques' in batch:
            print("aBot Validation:")

            # NOTE: A-Bot validation is slow, so adjust exampleLimit as needed
            rankMetrics = rankABot(aBot, dataset, 'val', scoringFunction=utils.maskedNll,
                                   exampleLimit=25 * params['batchSize'])

            for metric, value in rankMetrics.items():
                viz.linePlot(epochId, value, 'val - aBot', metric, xlabel='Epochs')

            if 'logProbsMean' in rankMetrics:
                logProbsMean = params['CELossCoeff'] * rankMetrics[
                    'logProbsMean']
                viz.linePlot(iterId, logProbsMean, 'aBotLoss', 'val CE')

                if params['trainMode'] == 'sl-abot':
                    valLoss = logProbsMean
                    viz.linePlot(iterId, valLoss, 'loss', 'val loss')

        if qBot:
            print("qBot Validation:")
            rankMetrics, roundMetrics = rankQBot(qBot, dataset, 'val')

            for metric, value in rankMetrics.items():
                print(iterId, epochId, value, 'val - qBot', metric)

            if 'logProbsMean' in rankMetrics:
                logProbsMean = params['CELossCoeff'] * rankMetrics[
                    'logProbsMean']
                viz.linePlot(iterId, logProbsMean, 'qBotLoss', 'val CE')

            if 'featLossMean' in rankMetrics:
                featLossMean = params['featLossCoeff'] * (
                    rankMetrics['featLossMean'])
                viz.linePlot(iterId, featLossMean, 'featLoss',
                             'val FeatureRegressionLoss')

            if 'logProbsMean' in rankMetrics and 'featLossMean' in rankMetrics:
                if params['trainMode'] == 'sl-qbot':
                    valLoss = logProbsMean + featLossMean
                    print(iterId, valLoss, 'loss', 'val loss')

    # ---------------------------------------------------------------------------
    # Save the model every epoch
    # ---------------------------------------------------------------------------
    if iterId % numIterPerEpoch == 0:
        params['ckpt_iterid'] = iterId
        params['ckpt_lRate'] = lRate
        if aBot:
            saveFile = os.path.join(params['savePath'], 'abot_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(aBot, optimizer, saveFile, params)
        if qBot:
            saveFile = os.path.join(params['savePath'], 'qbot_ep_%d.vd' % curEpoch)
            print('Saving model: ' + saveFile)
            utils.saveModel(qBot, optimizer, saveFile, params)
