"""
One simple generic script to launch all
kinds of jobs. These are:
- SL-Training for Abot and Qbot
- SL-Training for Abot Discriminators
- RL-Training for Abot and Qbot
- RL-Training for Abot, Qbot and Discriminator
"""
import os
import datetime
import subprocess

# Visdom Arguments
from time import gmtime, strftime

VISDOM_SERVER = 'http://asimo.cc.gatech.edu'
VISDOM_PORT = '7777'

# Global Directories
DATA_GLOBAL_DIRECTORIES = '/srv/share/vmurahari3/visdial-rl/data/'
CKPT_GLOBAL_DIRECTORIES = '/srv/share/vmurahari3/visdial-rl/checkpoints/'
CKPT_GLOBAL_DIRECTORIES_2 = '/srv/share2/vmurahari3/visdial-rl/checkpoints/'
LOG_GLOBAL_DIRECTORIES = '/srv/share/vmurahari3/visdial-rl/train_logs/'
DIALOG_GLOBAL_DIRECTORIES = '/srv/share/vmurahari3/visdial-rl/dialog/'
HUMAN_STUDY_DIRECTORIES = '/srv/share/vmurahari3/visdial-rl/human_study/'

# LOG_GLOBAL_DIRECTORIES = 'train_logs/'

# Dictionary for the training scripts
TRAIN_SCRIPTS = {
'SL-Abot': 'train.py',
'SL-Qbot': 'train.py',
'SL-Qbot-Multi-Round': 'train_multi_round_diversity.py',
'RL-Bots': 'train.py',
'RL-Bots-Diversity': 'train.py',
'RL-Bots-Future-Reward':'train_future_reward.py',
'RL-Bots-Actor-Critic':'train_actor_critic.py',
'RL-Bots-Multi-GPU':'train_multi_gpu.py'
}

# Path to Data
DATA_PATH = {
'v0.5': {
    'inputImg': 'v0.5_data/data_img.h5',
    'inputQues': 'v0.5_data/chat_processed_data.h5',
    'inputJson': 'v0.5_data/chat_processed_params.json',
    'inputQuesGencaps': 'v0.5_data/chat_processed_data_gencaps.h5',
    'cocoDir':'/srv/share/datasets/coco/images',
    'cocoInfo':'/srv/share/datasets/coco/coco.json'
},
'v1.0': {
    'inputImg': 'v1.0_data/data_img.h5',
    'inputQues': 'v1.0_data/chat_processed_data.h5',
    'inputJson': 'v1.0_data/chat_processed_params.json',
    'denseAnnotation': "v1.0_data/visdial_1.0_val_dense_annotations.json",
    'inputQuesGencaps': 'v1.0_data/chat_processed_data.h5',
    'cocoDir': '/srv/share/datasets/coco/images',
    'cocoInfo':'/srv/share/datasets/coco/coco.json'
}
}
# for 1.0 we are using ground truth captions for self talk. Therefore, we don't have a
# gencaps file.

def run_commands(commands):
    cur_time = '{date:%Y-%m-%d-%H:%M:%S}'.format(date=datetime.datetime.now())
    os.mkdir(os.path.join(LOG_GLOBAL_DIRECTORIES,cur_time))
    # dump all the params here
    params_file = os.path.join(LOG_GLOBAL_DIRECTORIES,cur_time, "params.txt")
    with open(params_file,"w+") as file:
        for cmd in commands:
            file.write(cmd + "\n")
    #call the sh script for submiting the jobs
    with open("log.txt","w") as f:
        subprocess.call(['sh','launch_jobs.sh', params_file], stdout=f)

def run_val_plot_jobs(data_split, start_epoch, end_epoch, eval_mode_list= 'ABotRank QBotRank QABotsRank',
                      abot_startfrom = None, qbot_startfrom = None,
                      use_huber_loss = 0,
                      use_cos_similarity_loss = 0,
                      use_actual_huber_loss = 0,
                      huber_loss_coeff = 5,
                      cos_similarity_loss_coeff = 5,
                      actual_huber_loss_coeff = 5,
                      additional_job_spec=None):

    ENV_NAME = data_split + '_' + 'val_plot_' + eval_mode_list.replace(' ',"_") \
                + 'ep_'+ str(start_epoch) + '_'+ str(end_epoch)
    train_script = 'val_plot.py'
    CMD_ARG = 'python ' + train_script + ' ' \
    '-inputImg ' + DATA_GLOBAL_DIRECTORIES + DATA_PATH[data_split]['inputImg'] + ' ' + \
    '-inputQues ' + DATA_GLOBAL_DIRECTORIES + DATA_PATH[data_split]['inputQues'] + ' ' + \
    '-inputJson ' + DATA_GLOBAL_DIRECTORIES + DATA_PATH[data_split]['inputJson'] + ' ' + \
    '-enableVisdom 1 ' + \
    '-cocoDir ' + DATA_PATH[data_split]['cocoDir'] + ' ' + \
    '-cocoInfo ' + DATA_PATH[data_split]['cocoInfo'] + ' ' + \
    '-visdomServer ' + VISDOM_SERVER + ' ' + \
    '-visdomServerPort ' + VISDOM_PORT + ' ' + \
    '-useGPU ' + \
    '-batchSize 20 ' + \
    '-evalMode ' + eval_mode_list + ' ' + \
    '-useNDCG ' + \
    '-startEpoch ' + str(start_epoch) + ' ' + \
    '-endEpoch ' + str(end_epoch) + ' '

    CMD_ARG += '-useHuberLoss ' + str(use_huber_loss) + ' ' \
            +'-HuberLossCoeff ' + str(huber_loss_coeff) + ' ' \
            + '-useSimilarityLoss ' + str(use_cos_similarity_loss) + ' ' \
            + '-SimilarityLossCoeff ' + str(cos_similarity_loss_coeff) + ' ' \
            + '-useActualHuberLoss ' + str(use_actual_huber_loss) + ' ' \
            + '-ActualHuberLossCoeff ' + str(actual_huber_loss_coeff) + ' '

    if data_split == "v1.0":
        CMD_ARG += ' -inputDenseJson ' + DATA_GLOBAL_DIRECTORIES + \
                   DATA_PATH[data_split]['denseAnnotation'] + ' '

    if ('ABotRank' in eval_mode_list) or ('QABotsRank' in eval_mode_list):
        CMD_ARG += '-startFrom ' + CKPT_GLOBAL_DIRECTORIES_2 + abot_startfrom + ' '

    if ('QBotRank' in eval_mode_list) or ('QABotsRank' in eval_mode_list):
        CMD_ARG += '-qstartFrom ' + CKPT_GLOBAL_DIRECTORIES_2 + qbot_startfrom + ' '

    if additional_job_spec != None:
        ENV_NAME += additional_job_spec + '_'

    timeStamp = strftime('%d-%b-%y-%X-%a', gmtime())

    ENV_NAME += 'job'
    CMD_ARG += '-visdomEnv ' + ENV_NAME + ' ' + '-saveName ' + timeStamp + ENV_NAME + ' '
    print(CMD_ARG)
    return CMD_ARG

def run_human_study(data_split,
                abot_startfrom,
                qbot_startfrom,
                beam_size=5,
                additional_job_spec=None):

    ENV_NAME = 'human-study' + data_split + '_' + '_beam_%d_'%beam_size
    CMD_ARG = 'python ' + 'evaluate.py' + ' ' \
    '-inputImg ' + DATA_GLOBAL_DIRECTORIES + DATA_PATH[data_split]['inputImg'] + ' ' + \
    '-inputQues ' + DATA_GLOBAL_DIRECTORIES + DATA_PATH[data_split]['inputQuesGencaps'] + ' ' + \
    '-inputJson ' + DATA_GLOBAL_DIRECTORIES + DATA_PATH[data_split]['inputJson'] + ' ' + \
    '-batchSize 20 '+ \
    '-evalMode human_study ' + \
    '-beamSize %d '%beam_size + \
    '-cocoDir ' + DATA_PATH[data_split]['cocoDir'] + ' ' + \
    '-cocoInfo ' + DATA_PATH[data_split]['cocoInfo'] + ' ' + '-useGPU '

    CMD_ARG += '-startFrom ' + CKPT_GLOBAL_DIRECTORIES_2 + abot_startfrom + ' ' + \
    '-qstartFrom ' + CKPT_GLOBAL_DIRECTORIES_2 + qbot_startfrom + ' '

    if additional_job_spec != None:
        ENV_NAME += additional_job_spec + '_'

    timeStamp = strftime('%d-%b-%y-%X-%a', gmtime())

    CMD_ARG += ' ' + '-savePath ' + HUMAN_STUDY_DIRECTORIES + \
               ' ' + '-saveName ' + timeStamp + ENV_NAME + ' '
    print(CMD_ARG)
    return CMD_ARG

def run_dialog(data_split,
                abot_startfrom,
                qbot_startfrom,
                beam_size=5,
                additional_job_spec=None):

    ENV_NAME = data_split + '_' + '_beam_%d_'%beam_size
    CMD_ARG = 'python ' + 'evaluate.py' + ' ' \
    '-inputImg ' + DATA_GLOBAL_DIRECTORIES + DATA_PATH[data_split]['inputImg'] + ' ' + \
    '-inputQues ' + DATA_GLOBAL_DIRECTORIES + DATA_PATH[data_split]['inputQuesGencaps'] + ' ' + \
    '-inputJson ' + DATA_GLOBAL_DIRECTORIES + DATA_PATH[data_split]['inputJson'] + ' ' + \
    '-batchSize 20 '+ \
    '-evalMode dialog ' + \
    '-beamSize %d '%beam_size + \
    '-cocoDir ' + DATA_PATH[data_split]['cocoDir'] + ' ' + \
    '-cocoInfo ' + DATA_PATH[data_split]['cocoInfo'] + ' ' + '-useGPU '

    CMD_ARG += '-startFrom ' + CKPT_GLOBAL_DIRECTORIES_2 + abot_startfrom + ' ' + \
    '-qstartFrom ' + CKPT_GLOBAL_DIRECTORIES_2 + qbot_startfrom + ' '

    if additional_job_spec != None:
        ENV_NAME += additional_job_spec + '_'

    timeStamp = strftime('%d-%b-%y-%X-%a', gmtime())

    CMD_ARG += ' ' + '-savePath ' + DIALOG_GLOBAL_DIRECTORIES + \
               ' ' + '-saveName ' + timeStamp + ENV_NAME + ' '
    print(CMD_ARG)
    return CMD_ARG

def run_evaluate(data_split,
                abot_startfrom,
                qbot_startfrom,
                beam_size=5,eval_mode_list= 'ABotRank QBotRank QABotsRank',
                additional_job_spec=None):

    ENV_NAME = data_split + '_' + 'val_plot_' + eval_mode_list.replace(' ',"_") \

    CMD_ARG = 'python ' + 'evaluate.py' + ' ' \
    '-inputImg ' + DATA_GLOBAL_DIRECTORIES + DATA_PATH[data_split]['inputImg'] + ' ' + \
    '-inputQues ' + DATA_GLOBAL_DIRECTORIES + DATA_PATH[data_split]['inputQuesGencaps'] + ' ' + \
    '-inputJson ' + DATA_GLOBAL_DIRECTORIES + DATA_PATH[data_split]['inputJson'] + ' ' + \
    '-batchSize 20 '+ \
    '-evalMode  ' + eval_mode_list + ' ' + \
    '-beamSize %d '%beam_size + \
    '-cocoDir ' + DATA_PATH[data_split]['cocoDir'] + ' ' + \
    '-cocoInfo ' + DATA_PATH[data_split]['cocoInfo'] + ' ' + '-useGPU ' + \
    '-enableVisdom 1 ' + \
    '-visdomServer ' + VISDOM_SERVER + ' ' + \
    '-visdomServerPort ' + VISDOM_PORT + ' ' + \
    '-useNDCG '

    if data_split == "v1.0":
        CMD_ARG += ' -inputDenseJson ' + DATA_GLOBAL_DIRECTORIES + \
                   DATA_PATH[data_split]['denseAnnotation'] + ' '

    if ('ABotRank' in eval_mode_list) or ('QABotsRank' in eval_mode_list):
        CMD_ARG += ' -startFrom ' + CKPT_GLOBAL_DIRECTORIES_2 + abot_startfrom + ' '

    if ('QBotRank' in eval_mode_list) or ('QABotsRank' in eval_mode_list):
        CMD_ARG += '-qstartFrom ' + CKPT_GLOBAL_DIRECTORIES_2 + qbot_startfrom + ' '

    if additional_job_spec != None:
        ENV_NAME += additional_job_spec + '_'

    timeStamp = strftime('%d-%b-%y-%X-%a', gmtime())

    ENV_NAME += 'job'

    CMD_ARG += ' ' + '-saveName ' + timeStamp + ENV_NAME + ' ' + '-visdomEnv ' + ENV_NAME
    print(CMD_ARG)
    return CMD_ARG

# Create cmd-arguments for SL/RL/RL-Disc jobs
def run_sl_rl_jobs(data_split,
                exp_mode,
                rl_abot_startfrom=None,
                rl_qbot_startfrom=None,
                rl_feat_loss_coeff=1000,
                rl_ce_loss_coeff=1,
                rl_loss_coeff = 2000,
                rl_curriculum_mode=1,
                rl_abot_reward=1,
                dropout = 0.5,
                use_huber_loss = 0,
                use_reconstruction_loss = 0,
                use_cos_similarity_loss = 0,
                use_actual_huber_loss = 0,
                huber_loss_coeff = 5,
                reconstruction_loss_coeff = 5,
                cos_similarity_loss_coeff = 5,
                actual_huber_loss_coeff = 5,
                annealing_end = 3,
                annealing_modulo = 1,
                additional_job_spec=None,use_continue=False,use_ndcg=True):

    ENV_NAME = exp_mode + '_' + data_split + '_' + 'drop_' + str(dropout) + "_"
    train_script = TRAIN_SCRIPTS[exp_mode]
    CMD_ARG = 'python ' + train_script + ' ' \
    '-inputImg ' + DATA_GLOBAL_DIRECTORIES + DATA_PATH[data_split]['inputImg'] + ' ' + \
    '-inputQues ' + DATA_GLOBAL_DIRECTORIES + DATA_PATH[data_split]['inputQues'] + ' ' + \
    '-inputJson ' + DATA_GLOBAL_DIRECTORIES + DATA_PATH[data_split]['inputJson'] + ' ' + \
    '-cocoDir ' + DATA_PATH[data_split]['cocoDir'] + ' ' + \
    '-cocoInfo ' + DATA_PATH[data_split]['cocoInfo'] + ' ' + \
    '-enableVisdom 1 ' + \
    '-visdomServer ' + VISDOM_SERVER + ' ' + \
    '-visdomServerPort ' + VISDOM_PORT + ' ' + \
    '-dropout ' + str(dropout) + ' ' + \
    '-useGPU ' + \
    '-batchSize 20 ' + \
    '-learningRate 1e-3 ' + \
    '-annealingEndRound ' + str(annealing_end) + " " + \
    '-annealingReduceEpoch ' + str(annealing_modulo) + " " 

    if use_continue:
        CMD_ARG += ' -continue '

    if use_ndcg:
        CMD_ARG += ' -useNDCG '

    if data_split == 'v1.0':
        CMD_ARG +=  ' -inputDenseJson /srv/share/vmurahari3/visdial-rl/data/v1.0_data/visdial_1.0_val_dense_annotations.json '

    if 'SL' in exp_mode:
        ENV_NAME += exp_mode + '_'
        ENV_NAME += str(rl_ce_loss_coeff) + '_cel_'

        if 'Qbot' in exp_mode:
            ENV_NAME += str(rl_feat_loss_coeff) + '_ftl_'
            CMD_ARG += '-trainMode sl-qbot ' + \
            '-featLossCoeff ' + str(rl_feat_loss_coeff) + ' ' + \
            '-CELossCoeff ' + str(rl_ce_loss_coeff) + ' '
            if rl_qbot_startfrom != None:
                CMD_ARG += '-qstartFrom ' + CKPT_GLOBAL_DIRECTORIES_2 + rl_qbot_startfrom + ' '

        else:
            CMD_ARG += '-trainMode sl-abot ' + \
            '-CELossCoeff ' + str(rl_ce_loss_coeff) + ' '
            if rl_abot_startfrom != None:
                CMD_ARG += '-startFrom ' + CKPT_GLOBAL_DIRECTORIES_2 + rl_abot_startfrom + ' '

    elif 'RL-Bots' in exp_mode:
        ENV_NAME += 'SLABotRL_' + exp_mode + '_'
        ENV_NAME += str(rl_ce_loss_coeff) + '_cel_'
        ENV_NAME += str(rl_feat_loss_coeff) + '_ftl_'
        ENV_NAME += str(rl_loss_coeff) + '_rlc_'
        ENV_NAME += 'anneal_' + \
                  str(annealing_end) + '_' + str(annealing_modulo) + '_'
        CMD_ARG += '-startFrom ' + CKPT_GLOBAL_DIRECTORIES_2 + rl_abot_startfrom + ' ' + \
        '-qstartFrom ' + CKPT_GLOBAL_DIRECTORIES_2 + rl_qbot_startfrom + ' ' + \
        '-RLLossCoeff ' + str(rl_loss_coeff) + ' ' + \
        '-featLossCoeff ' + str(rl_feat_loss_coeff) + ' ' + \
        '-CELossCoeff ' + str(rl_ce_loss_coeff) + ' ' + \
        '-rlAbotReward ' + str(rl_abot_reward) + ' ' + \
        '-useCurriculum ' + str(rl_curriculum_mode) + ' ' + \
        '-trainMode rl-full-QAf '

        # if 'RL-Bots-Future-Reward' or 'RL-Bots-Actor-Critic' in exp_mode:
        #     ENV_NAME += 'disc_' + str(discount)
        # if 'RL-Bots-Multi-GPU' in exp_mode:
        #     ENV_NAME += 'batchMutliply_' + str(batchMultiply)

    # Add arguments for jobs with diversity penalty
    CMD_ARG += '-useHuberLoss ' + str(use_huber_loss) + ' ' \
            +'-HuberLossCoeff ' + str(huber_loss_coeff) + ' ' \
            + '-useCosSimilarityLoss ' + str(use_cos_similarity_loss) + ' ' \
            + '-CosSimilarityLossCoeff ' + str(cos_similarity_loss_coeff) + ' ' \

    if use_huber_loss:
        ENV_NAME += "Huber_" + str(huber_loss_coeff)
    if use_cos_similarity_loss:
        ENV_NAME += "Cos_" + str(cos_similarity_loss_coeff)

    if additional_job_spec != None:
        ENV_NAME += additional_job_spec + '_'

    timeStamp = strftime('%d-%b-%y-%X-%a', gmtime())

    ENV_NAME += 'job'

    if use_continue:
        ENV_NAME +='continue'

    CMD_ARG += '-visdomEnv ' + ENV_NAME + ' ' + '-saveName ' + timeStamp + ENV_NAME + ' '
    print(CMD_ARG)
    return CMD_ARG

commands = []

aBot_epoch = "09-Feb-19-06:09:46-Sat_1689132/abot_ep_40.vd"
qBot_epoch = "09-Feb-19-06:29:48-Sat_7609112/qbot_ep_18.vd"

# commands.append(run_sl_rl_jobs("v1.0", "SL-Qbot", dropout=0, additional_job_spec="release_baseline"))
# commands.append(run_sl_rl_jobs("v1.0", "SL-Abot", dropout=0, additional_job_spec="release_baseline"))

cmd = run_sl_rl_jobs("v1.0", "RL-Bots",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, rl_feat_loss_coeff=10000, rl_ce_loss_coeff=100, annealing_end=3, dropout=0,additional_job_spec="ICCVRLReleaseNoBaseline")
commands.append(cmd)

'''
HuberLossCoeffs = [2,2.5]
for hlc in HuberLossCoeffs:
    commands.append(run_sl_rl_jobs("v1.0","SL-Qbot",
                   use_huber_loss=1,huber_loss_coeff=hlc))

commands.append(run_sl_rl_jobs("v1.0","SL-Abot"))
#write command to a file
commands.append(run_sl_rl_jobs("v1.0","SL-Qbot"))
'''
'''
HuberLossCoeffs = [2,2.5,3,3.5,4,4.5,5,6]
save_path_suffix=["09-Feb-19-06:09:25-Sat_3404123","09-Feb-19-06:09:25-Sat_5379076","08-Feb-19-23:24:39-Fri_4083540"
                  ,"08-Feb-19-23:24:39-Fri_7111058","08-Feb-19-23:24:39-Fri_172501",
                  "08-Feb-19-23:24:39-Fri_9391277","08-Feb-19-23:24:39-Fri_534450",
                  "08-Feb-19-23:24:39-Fri_150551"]

best_epoch_numbers = [21, 28, 26, 26, 30, 26, 28, 26]
'''

'''
HuberLossCoeffs = [4.5]
best_epoch_numbers = [64]
save_path_suffix=["08-Feb-19-23:24:39-Fri_9391277"]

for ind in range(len(HuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=os.path.join(save_path_suffix[ind],"qbot_ep_%d.vd"%best_epoch_numbers[ind]),
        rl_loss_coeff=1,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=200,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_EWA%s"%str(HuberLossCoeffs[ind]))
    commands.append(cmd)

for ind in range(len(HuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=os.path.join(save_path_suffix[ind],"qbot_ep_%d.vd"%best_epoch_numbers[ind]),
        rl_loss_coeff=10,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_EWA%s"%str(HuberLossCoeffs[ind]))
    commands.append(cmd)

for ind in range(len(HuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=os.path.join(save_path_suffix[ind],"qbot_ep_%d.vd"%best_epoch_numbers[ind]),
        rl_loss_coeff=1,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_EWA%s"%str(HuberLossCoeffs[ind]))
    commands.append(cmd)

for ind in range(len(HuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=os.path.join(save_path_suffix[ind],"qbot_ep_%d.vd"%best_epoch_numbers[ind]),
        rl_loss_coeff=100,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=200,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_EWA%s"%str(HuberLossCoeffs[ind]))
    commands.append(cmd)

for ind in range(len(HuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=os.path.join(save_path_suffix[ind],"qbot_ep_%d.vd"%best_epoch_numbers[ind]),
        rl_loss_coeff=20000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=200,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_EWA%s"%str(HuberLossCoeffs[ind]))
    commands.append(cmd)


for ind in range(len(HuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=os.path.join(save_path_suffix[ind],"qbot_ep_%d.vd"%best_epoch_numbers[ind]),
        rl_loss_coeff=20000,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_EWA%s"%str(HuberLossCoeffs[ind]))
    commands.append(cmd)

for ind in range(len(HuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=os.path.join(save_path_suffix[ind],"qbot_ep_%d.vd"%best_epoch_numbers[ind]),
        rl_loss_coeff=2000,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_EWA%s"%str(HuberLossCoeffs[ind]))
    commands.append(cmd)

for ind in range(len(HuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=os.path.join(save_path_suffix[ind],"qbot_ep_%d.vd"%best_epoch_numbers[ind]),
        rl_loss_coeff=8000,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_EWA%s"%str(HuberLossCoeffs[ind]))
    commands.append(cmd)

for ind in range(len(HuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=os.path.join(save_path_suffix[ind],"qbot_ep_%d.vd"%best_epoch_numbers[ind]),
        rl_loss_coeff=1,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_EWA%s"%str(HuberLossCoeffs[ind]))
    commands.append(cmd)
'''

'''
ActualHuberLossCoeffs = [0.1]
checkpt ="26-Feb-19-08:07:02-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.1Finetune_ep78_job/qbot_ep_130.vd"

for ind in range(len(ActualHuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=checkpt,
        rl_loss_coeff=1,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="ActualHuber_Pretrain_ep130_EWA%s"%str(ActualHuberLossCoeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(ActualHuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=checkpt,
        rl_loss_coeff=20000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=200,annealing_end=3,dropout=0,additional_job_spec="ActualHuber_Pretrain_ep130_EWA%s"%str(ActualHuberLossCoeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(ActualHuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=checkpt,
        rl_loss_coeff=1,annealing_end=3,dropout=0,additional_job_spec="ActualHuber_Pretrain_ep130_EWA%s"%str(ActualHuberLossCoeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(ActualHuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=checkpt,
        rl_loss_coeff=1,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=4,dropout=0,additional_job_spec="ActualHuber_Pretrain_ep130_EWA%s"%str(ActualHuberLossCoeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(ActualHuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=checkpt,
        rl_loss_coeff=20000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=200,annealing_end=4,dropout=0,additional_job_spec="ActualHuber_Pretrain_ep130_EWA%s"%str(ActualHuberLossCoeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(ActualHuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=checkpt,
        rl_loss_coeff=1,annealing_end=4,dropout=0,additional_job_spec="ActualHuber_Pretrain_ep130_EWA%s"%str(ActualHuberLossCoeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(ActualHuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=checkpt,
        rl_loss_coeff=1,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=5,dropout=0,additional_job_spec="ActualHuber_Pretrain_ep130_EWA%s"%str(ActualHuberLossCoeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(ActualHuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=checkpt,
        rl_loss_coeff=20000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=200,annealing_end=5,dropout=0,additional_job_spec="ActualHuber_Pretrain_ep130_EWA%s"%str(ActualHuberLossCoeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(ActualHuberLossCoeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=aBot_epoch,
        rl_qbot_startfrom=checkpt,
        rl_loss_coeff=1,annealing_end=5,dropout=0,additional_job_spec="ActualHuber_Pretrain_ep130_EWA%s"%str(ActualHuberLossCoeffs[ind])
                        )
    commands.append(cmd)
'''

'''
HuberLossCoeffs = [2,2.5,3,3.5,4,4.5,5,6]
Rl_dirs = ["10-Feb-19-19:45:39-Sunv1.0_RL-Bots_1_cel_1000_ftl_2000_rlc_Huber_Pretrain_2_job",
           '10-Feb-19-19:45:39-Sunv1.0_RL-Bots_1_cel_1000_ftl_2000_rlc_Huber_Pretrain_2.5_job',
           '10-Feb-19-19:45:39-Sunv1.0_RL-Bots_1_cel_1000_ftl_2000_rlc_Huber_Pretrain_3_job',
           "10-Feb-19-19:45:39-Sunv1.0_RL-Bots_1_cel_1000_ftl_2000_rlc_Huber_Pretrain_3.5_job",
           "10-Feb-19-19:45:39-Sunv1.0_RL-Bots_1_cel_1000_ftl_2000_rlc_Huber_Pretrain_4_job",
           "10-Feb-19-19:45:39-Sunv1.0_RL-Bots_1_cel_1000_ftl_2000_rlc_Huber_Pretrain_4.5_job",
           "10-Feb-19-19:45:39-Sunv1.0_RL-Bots_1_cel_1000_ftl_2000_rlc_Huber_Pretrain_5_job",
           "10-Feb-19-19:45:39-Sunv1.0_RL-Bots_1_cel_1000_ftl_2000_rlc_Huber_Pretrain_6_job"]

for ind in range(len(HuberLossCoeffs)):
    cmd = run_val_plot_jobs("v1.0", 1, 35, eval_mode_list='ABotRank QBotRank QABotsRank', abot_startfrom= Rl_dirs[ind],
        qbot_startfrom= Rl_dirs[ind], additional_job_spec="RL_2000_Huber_%s"%str(HuberLossCoeffs[ind]))
    commands.append(cmd)
'''

'''
commands.append(run_val_plot_jobs("v1.0", 40, 41, eval_mode_list='ABotRank', abot_startfrom="09-Feb-19-06:09:46-Sat_1689132",
         qbot_startfrom="09-Feb-19-06:09:46-Sat_1689132", additional_job_spec="Abot_baseline"))
'''
'''
commands.append(run_val_plot_jobs("v1.0", 1, 60, eval_mode_list='ABotRank', abot_startfrom="06-Mar-19-05:40:09-Wedv1.0_drop_0_SL-Abot_1_cel_ActualHuber_0.001Finetune_job",
         qbot_startfrom="06-Mar-19-05:40:09-Wedv1.0_drop_0_SL-Abot_1_cel_ActualHuber_0.001Finetune_job",additional_job_spec="06-Mar-19-05:40:09-Wedv1.0_drop_0_SL-Abot_1_cel_ActualHuber_0.001Finetune_job"))
commands.append(run_val_plot_jobs("v1.0", 1, 60, eval_mode_list='ABotRank', abot_startfrom="06-Mar-19-05:40:09-Wedv1.0_drop_0_SL-Abot_1_cel_ActualHuber_0.01Finetune_job",
         qbot_startfrom="06-Mar-19-05:40:09-Wedv1.0_drop_0_SL-Abot_1_cel_ActualHuber_0.01Finetune_job",additional_job_spec="06-Mar-19-05:40:09-Wedv1.0_drop_0_SL-Abot_1_cel_ActualHuber_0.01Finetune_job"))
commands.append(run_val_plot_jobs("v1.0", 1, 60, eval_mode_list='ABotRank', abot_startfrom="06-Mar-19-05:40:09-Wedv1.0_drop_0_SL-Abot_1_cel_ActualHuber_0.1Finetune_job",
         qbot_startfrom="06-Mar-19-05:40:09-Wedv1.0_drop_0_SL-Abot_1_cel_ActualHuber_0.1Finetune_job",additional_job_spec="06-Mar-19-05:40:09-Wedv1.0_drop_0_SL-Abot_1_cel_ActualHuber_0.1Finetune_job"))
'''
# commands.append(run_val_plot_jobs("v1.0", 4, 5, eval_mode_list='ABotRank', abot_startfrom="13-May-19-06:14:15-MonRL-Bots_v1.0_drop_0_RL-Bots_40_cel_1000_ftl_100_rlc_anneal_3_1_disc_0.5ActualHuberFinetune_0.07_job",
#          qbot_startfrom="13-May-19-06:14:15-MonRL-Bots_v1.0_drop_0_RL-Bots_40_cel_1000_ftl_100_rlc_anneal_3_1_disc_0.5ActualHuberFinetune_0.07_job",additional_job_spec="test"))

# commands.append(run_val_plot_jobs("v1.0", 4, 5, eval_mode_list='ABotRank', abot_startfrom="13-May-19-06:14:15-MonRL-Bots_v1.0_drop_0_RL-Bots_40_cel_1000_ftl_100_rlc_anneal_3_1_disc_0.5ActualHuberFinetune_0.07_job",
#          qbot_startfrom="13-May-19-06:14:15-MonRL-Bots_v1.0_drop_0_RL-Bots_40_cel_1000_ftl_100_rlc_anneal_3_1_disc_0.5ActualHuberFinetune_0.07_job",additional_job_spec="test"))

# commands.append(run_val_plot_jobs("v1.0", 6,7, eval_mode_list='ABotRank', abot_startfrom="09-May-19-14:58:47-Thuv1.0_drop_0_SL-Abot_1_cel_ActualHuber_5e-05Finetune_job",
#          qbot_startfrom="09-May-19-14:58:47-Thuv1.0_drop_0_SL-Abot_1_cel_ActualHuber_5e-05Finetune_job",additional_job_spec="test"))

'''
'''

abot_huber_diverse = "09-May-19-14:58:47-Thuv1.0_drop_0_SL-Abot_1_cel_ActualHuber_5e-05Finetune_job/abot_ep_6.vd"

'''
commands.append(run_evaluate("v1.0", abot_startfrom=aBot_epoch,
         qbot_startfrom=qBot_epoch,eval_mode_list='QABotsRank',additional_job_spec="CE-Qbot-CE-Abot"))

commands.append(run_evaluate("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom='01-May-19-04:23:08-Wedv1.0_drop_0.5_SL-Qbot_0.1_cel_1000_ftl_ActualHuber_0.007finetune_job/qbot_ep_75.vd',
                             eval_mode_list='QABotsRank',additional_job_spec='DivQbot-CEAbot',beam_size=5))

commands.append(run_evaluate("v1.0",abot_startfrom=abot_huber_diverse,
                            qbot_startfrom='01-May-19-04:23:08-Wedv1.0_drop_0.5_SL-Qbot_0.1_cel_1000_ftl_ActualHuber_0.007finetune_job/qbot_ep_75.vd'
                             ,eval_mode_list='QABotsRank',additional_job_spec='DivQbot-DivAbot',beam_size=5))

commands.append(run_evaluate("v1.0",abot_startfrom='13-Mar-19-01:34:58-Wedv1.0_drop_0_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_job/abot_ep_4.vd',
                            qbot_startfrom='13-Mar-19-01:34:58-Wedv1.0_drop_0_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_job/qbot_ep_4.vd',
                             eval_mode_list='QABotsRank',additional_job_spec='ICCV-RLQbot-ICCV-RLAbot',beam_size=5))

commands.append(run_evaluate("v1.0",abot_startfrom='15-May-19-15:44:49-WedRL-Bots_v1.0_drop_0_SLABotRL_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_disc_0.5Chkpt2_ActualHuberFinetune_0.07_job/abot_ep_8.vd',
                            qbot_startfrom='15-May-19-15:44:49-WedRL-Bots_v1.0_drop_0_SLABotRL_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_disc_0.5Chkpt2_ActualHuberFinetune_0.07_job/qbot_ep_8.vd',
                             eval_mode_list='QABotsRank',additional_job_spec='RLQbot-RLAbot-Finetuned-CEAbot',beam_size=5))

commands.append(run_evaluate("v1.0",abot_startfrom='13-May-19-06:14:15-MonRL-Bots_v1.0_drop_0_RL-Bots_40_cel_1000_ftl_100_rlc_anneal_3_1_disc_0.5ActualHuberFinetune_0.07_job/abot_ep_4.vd',
                            qbot_startfrom='13-May-19-06:14:15-MonRL-Bots_v1.0_drop_0_RL-Bots_40_cel_1000_ftl_100_rlc_anneal_3_1_disc_0.5ActualHuberFinetune_0.07_job/qbot_ep_4.vd',
                             eval_mode_list='QABotsRank',additional_job_spec='RLQbot-RLAbot-Finetuned-DivAbot',beam_size=5))
'''
# commands.append(run_evaluate("v1.0",abot_startfrom=abot_huber_diverse,
#                             qbot_startfrom=qBot_epoch,
#                              eval_mode_list='QABotsRank',additional_job_spec='CEQbot-DivAbot',beam_size=5))


'''
huber_coeffs=[3.5,4,4.5]
# continuing past job to run it for more epochs
abot_dirs = ['13-Feb-19-19:11:22-Wedv1.0_drop_0.5_SL-Abot_1_cel_Huber_3.5job/abot_ep_64.vd',
        '13-Feb-19-19:11:22-Wedv1.0_drop_0.5_SL-Abot_1_cel_Huber_4job/abot_ep_64.vd',
        '13-Feb-19-19:11:22-Wedv1.0_drop_0.5_SL-Abot_1_cel_Huber_4.5job/abot_ep_64.vd']
for ind, hlc in enumerate(huber_coeffs):
    commands.append(run_sl_rl_jobs("v1.0","SL-Abot",use_huber_loss=1,rl_abot_startfrom=abot_dirs[ind],
                                   huber_loss_coeff=hlc,additional_job_spec="continue_from_ep65"))
'''

# running cos similarity diversity SL Qbot on 1.0
'''
Cos_Similarity_Loss_Coeff = [0.1,0.5,1,2,5,10,50]
for coeff in Cos_Similarity_Loss_Coeff:
    commands.append(run_sl_rl_jobs("v1.0","SL-Qbot",rl_qbot_startfrom="08-Feb-19-23:24:39-Fri_9391277/qbot_ep_26.vd",
                   use_cos_similarity_loss=1,cos_similarity_loss_coeff=coeff,
                                   additional_job_spec="Finetune_FrozenDecoder"))
'''
# running actual huber loss SL Qbot on 1.0
'''
Actual_Huber_Loss_Coeff = [0.01,0.1,0.5,2,5,50]
for coeff in Actual_Huber_Loss_Coeff:
    commands.append(run_sl_rl_jobs("v1.0","SL-Qbot",rl_qbot_startfrom=qBot_epoch,
                   use_actual_huber_loss=1,actual_huber_loss_coeff=coeff,additional_job_spec="Finetune"))
'''
'''
Cos_Loss_Coeff = [1]
for coeff in Cos_Loss_Coeff:
    commands.append(run_sl_rl_jobs("v1.0","SL-Abot",rl_abot_startfrom=aBot_epoch,
                   use_cos_similarity_loss=1,dropout=0,cos_similarity_loss_coeff=coeff,additional_job_spec="Finetune"))
'''
'''
Huber_Loss_Coeff = [.00001,.0001,.001,0.01,0.1,1]
for coeff in Huber_Loss_Coeff:
    commands.append(run_sl_rl_jobs("v1.0","SL-Abot",rl_abot_startfrom=aBot_epoch,
                   use_huber_loss=1,dropout=0,huber_loss_coeff=coeff,additional_job_spec="Finetune"))
'''

# Actual_Huber_Loss_Coeff = [0.00005,0.0001,0.0005,.001,.005]
# for coeff in Actual_Huber_Loss_Coeff:
#     commands.append(run_sl_rl_jobs("v1.0","SL-Abot",rl_abot_startfrom=aBot_epoch,
#                    use_actual_huber_loss=1,dropout=0,actual_huber_loss_coeff=coeff,additional_job_spec="Finetune"))
#

# commands.append(run_sl_rl_jobs("v1.0","SL-Qbot",rl_qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.1Finetune_job"
#                                            "/qbot_ep_78.vd",
#                                use_actual_huber_loss=1,actual_huber_loss_coeff=0.1,additional_job_spec="Finetune_ep78"))

'''
Actual_Huber_Loss_Coeff = [0.1,0.5,1,2,5,10,50]
for coeff in Actual_Huber_Loss_Coeff:
    commands.append(run_sl_rl_jobs("v1.0","SL-Qbot",use_actual_huber_loss=1,actual_huber_loss_coeff=coeff))
'''
# cos + actual huber
'''
Actual_Huber_Loss_Coeff = [0.1,0.5,1,2,5,10,50]
for coeff in Actual_Huber_Loss_Coeff:
    commands.append(run_sl_rl_jobs("v1.0","SL-Qbot",rl_qbot_startfrom="08-Feb-19-23:24:39-Fri_9391277/qbot_ep_26.vd",
                   use_actual_huber_loss=1,use_cos_similarity_loss=1
                 ,actual_huber_loss_coeff=coeff,cos_similarity_loss_coeff=coeff,additional_job_spec="Finetune_FrozenDecoder"))
'''
'''
# cos + Huber
Huber_Loss_Coeff = [0.1,0.5,1,2,5,10,50]
for coeff in Huber_Loss_Coeff:
    commands.append(run_sl_rl_jobs("v1.0","SL-Qbot",rl_qbot_startfrom="08-Feb-19-23:24:39-Fri_9391277/qbot_ep_26.vd",
                   use_huber_loss=1,use_cos_similarity_loss=1
                 ,huber_loss_coeff=coeff,cos_similarity_loss_coeff=coeff,additional_job_spec="Finetune_FrozenDecoder"))
'''
'''
#sanity check Actual Huber
commands.append(run_sl_rl_jobs("v1.0","SL-Qbot",rl_qbot_startfrom="08-Feb-19-23:24:39-Fri_9391277/qbot_ep_26.vd",
                   use_actual_huber_loss=1,rl_ce_loss_coeff=0,actual_huber_loss_coeff=1))
commands.append(run_sl_rl_jobs("v1.0","SL-Qbot",rl_qbot_startfrom="08-Feb-19-23:24:39-Fri_9391277/qbot_ep_26.vd",
                   use_actual_huber_loss=1,dropout=0,actual_huber_loss_coeff=1))
'''

'''
Actual_Huber_Loss_Coeff = [1000000,1000000,100000000,1000000000]
for coeff in Actual_Huber_Loss_Coeff:
    commands.append(run_sl_rl_jobs("v1.0","SL-Qbot",use_actual_huber_loss=1,actual_huber_loss_coeff=coeff))
'''
# running cos RL jobs
'''
cos_qbot_checkpt = "16-Feb-19-19:52:56-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_2job/qbot_ep_47.vd"

cmd =run_sl_rl_jobs("v1.0","RL-Bots",
    rl_abot_startfrom=aBot_epoch,
    rl_qbot_startfrom=cos_qbot_checkpt,
    rl_loss_coeff=20000,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=200,annealing_end=3,dropout=0,additional_job_spec="Cos")
commands.append(cmd)

cmd =run_sl_rl_jobs("v1.0","RL-Bots",
    rl_abot_startfrom=aBot_epoch,
    rl_qbot_startfrom=cos_qbot_checkpt,
    rl_loss_coeff=200,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,additional_job_spec="Cos")
commands.append(cmd)

cmd =run_sl_rl_jobs("v1.0","RL-Bots",
    rl_abot_startfrom=aBot_epoch,
    rl_qbot_startfrom=cos_qbot_checkpt,
    rl_loss_coeff=20000,annealing_end=3,dropout=0,additional_job_spec="Cos")
commands.append(cmd)

cmd =run_sl_rl_jobs("v1.0","RL-Bots",
    rl_abot_startfrom=aBot_epoch,
    rl_qbot_startfrom=cos_qbot_checkpt,
    rl_loss_coeff=1,annealing_end=3,dropout=0,additional_job_spec="Cos")
commands.append(cmd)

cmd =run_sl_rl_jobs("v1.0","RL-Bots",
    rl_abot_startfrom=aBot_epoch,
    rl_qbot_startfrom=cos_qbot_checkpt,
    rl_loss_coeff=1,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=200,annealing_end=3,dropout=0,additional_job_spec="Cos")
commands.append(cmd)
'''

# Dialog Jobs

'''
commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="09-Feb-19-06:29:48-Sat_7609112/qbot_ep_59.vd"
                            ,additional_job_spec="SLQbot_Pure_ep59",beam_size=1))
commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="09-Feb-19-06:29:48-Sat_7609112/qbot_ep_59.vd"
                            ,additional_job_spec="SLQbot_Pure_ep59",beam_size=5))

'''
'''
commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,

                            qbot_startfrom="16-Feb-19-19:52:56-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_2job/qbot_ep_84.vd"
                            ,additional_job_spec="SLCos_ep84",beam_size=1))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="08-Feb-19-23:24:39-Fri_9391277/qbot_ep_64.vd"
                            ,additional_job_spec="SL_Huber_4.5_ep64",beam_size=1))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.5job/qbot_ep_72.vd"
                            ,additional_job_spec="SLActualHuber_ep72",beam_size=1))
'''

'''
commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="09-Apr-19-02:35:47-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Huber_0.0001finetune_job/qbot_ep_84.vd"
                            ,additional_job_spec="SL_Huber_finetune_1e-4_ep84",beam_size=5))
commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="09-Apr-19-02:35:47-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Huber_0.0001finetune_job/qbot_ep_84.vd"
                            ,additional_job_spec="SL_Huber_finetune_1e-4_ep84",beam_size=1))
'''
'''
commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="09-Apr-19-02:35:47-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_0.01finetune_job/qbot_ep_64.vd"
                            ,additional_job_spec="SL_Cos_finetune_1e-4_ep64",beam_size=1))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="09-Apr-19-02:35:47-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.001finetune_job/qbot_ep_60.vd"
                            ,additional_job_spec="SL_Actual_Huber_1e-3_ep_60",beam_size=1))
'''
#
# commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
#                             qbot_startfrom="23-Feb-19-23:21:05-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Huber_50Cos_50Finetune_FrozenDecoder_job"
#                                            "/qbot_ep_2.vd"
#                             ,additional_job_spec="SL_Huber_Cos_50_ep_2"))
# commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
#                             qbot_startfrom="23-Feb-19-23:21:05-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Huber_50Cos_50Finetune_FrozenDecoder_job"
#                                            "/qbot_ep_3.vd"
#                             ,additional_job_spec="SL_Huber_Cos_50_ep_3"))
# commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
#                             qbot_startfrom="23-Feb-19-23:21:05-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Huber_50Cos_50Finetune_FrozenDecoder_job"
#                                            "/qbot_ep_4.vd"
#                             ,additional_job_spec="SL_Huber_Cos_50_ep_4"))


# commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
#                             qbot_startfrom="08-Feb-19-23:24:39-Fri_9391277/qbot_ep_28.vd"
#                             ,additional_job_spec="SLHuber_28"))
#
# commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
#                             qbot_startfrom="08-Feb-19-23:24:39-Fri_9391277/qbot_ep_27.vd"
#                             ,additional_job_spec="SLHuber_27"))
# commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
#                             qbot_startfrom="08-Feb-19-23:24:39-Fri_9391277/qbot_ep_26.vd"
#                             ,additional_job_spec="SLHuber_ep_26_orig"))
# commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
#                             qbot_startfrom="08-Feb-19-23:24:39-Fri_9391277/qbot_ep_25.vd"
#                             ,additional_job_spec="SLHuber_ep25"))

# commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
#
#                             qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.5job/qbot_ep_72.vd"
#                             ,additional_job_spec="SLActualHuber_ep72"))

#
# commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
#                             qbot_startfrom="16-Feb-19-19:52:56-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_2job/qbot_ep_70.vd"
#                             ,additional_job_spec="SLCos_ep70"))

# commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
#                             qbot_startfrom="26-Feb-19-08:07:02-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.1Finetune_ep78_job"
#                                            "/qbot_ep_120.vd"
#                             ,additional_job_spec="SLActualHuber_Coeff_0.1_ep120"))
#

abot_huber_diverse = "09-May-19-14:58:47-Thuv1.0_drop_0_SL-Abot_1_cel_ActualHuber_5e-05Finetune_job/abot_ep_6.vd"



'''
commands.append(run_human_study("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom=qBot_epoch,additional_job_spec='GTQuestionsAnswers',beam_size=5))


commands.append(run_human_study("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom='01-May-19-04:23:08-Wedv1.0_drop_0.5_SL-Qbot_0.1_cel_1000_ftl_ActualHuber_0.007finetune_job/qbot_ep_75.vd',
                              additional_job_spec='DivQbot-CEAbot',beam_size=5))

commands.append(run_human_study("v1.0",abot_startfrom=abot_huber_diverse,
                            qbot_startfrom='01-May-19-04:23:08-Wedv1.0_drop_0.5_SL-Qbot_0.1_cel_1000_ftl_ActualHuber_0.007finetune_job/qbot_ep_75.vd'
                           ,additional_job_spec='DivQbot-DivAbot',beam_size=5))

commands.append(run_human_study("v1.0",abot_startfrom=abot_huber_diverse,
                            qbot_startfrom=qBot_epoch,
                           additional_job_spec='CEQbot-DivAbot',beam_size=5))

commands.append(run_human_study("v1.0",abot_startfrom='13-Mar-19-01:34:58-Wedv1.0_drop_0_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_job/abot_ep_4.vd',
                            qbot_startfrom='13-Mar-19-01:34:58-Wedv1.0_drop_0_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_job/qbot_ep_4.vd',
                           additional_job_spec='ICCV-RLQbot-ICCV-RLAbot',beam_size=5))

commands.append(run_human_study("v1.0",abot_startfrom='15-May-19-15:44:49-WedRL-Bots_v1.0_drop_0_SLABotRL_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_disc_0.5Chkpt2_ActualHuberFinetune_0.07_job/abot_ep_8.vd',
                            qbot_startfrom='15-May-19-15:44:49-WedRL-Bots_v1.0_drop_0_SLABotRL_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_disc_0.5Chkpt2_ActualHuberFinetune_0.07_job/qbot_ep_8.vd',
                           additional_job_spec='RLQbot-RLAbot-Finetuned-CEAbot',beam_size=5))

commands.append(run_human_study("v1.0",abot_startfrom='13-May-19-06:14:15-MonRL-Bots_v1.0_drop_0_RL-Bots_40_cel_1000_ftl_100_rlc_anneal_3_1_disc_0.5ActualHuberFinetune_0.07_job/abot_ep_4.vd',
                            qbot_startfrom='13-May-19-06:14:15-MonRL-Bots_v1.0_drop_0_RL-Bots_40_cel_1000_ftl_100_rlc_anneal_3_1_disc_0.5ActualHuberFinetune_0.07_job/qbot_ep_4.vd',
                           additional_job_spec='RLQbot-RLAbot-Finetuned-DivAbot',beam_size=5))
'''
'''
commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom='01-May-19-04:23:08-Wedv1.0_drop_0.5_SL-Qbot_0.1_cel_1000_ftl_ActualHuber_0.007finetune_job/qbot_ep_75.vd',
                              additional_job_spec='DivQbot-CEAbot',beam_size=5))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom=qBot_epoch,additional_job_spec='CEQbot-CEAbot',beam_size=5))

commands.append(run_dialog("v1.0",abot_startfrom='13-Mar-19-01:34:58-Wedv1.0_drop_0_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_job/abot_ep_4.vd',
                            qbot_startfrom='13-Mar-19-01:34:58-Wedv1.0_drop_0_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_job/qbot_ep_4.vd',
                           additional_job_spec='ICCV-RLQbot-ICCV-RLAbot',beam_size=5))

'''
'''
commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom=qBot_epoch,additional_job_spec='CEQbot-CEAbot',beam_size=5))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom='01-May-19-04:23:08-Wedv1.0_drop_0.5_SL-Qbot_0.1_cel_1000_ftl_ActualHuber_0.007finetune_job/qbot_ep_75.vd',
                              additional_job_spec='DivQbot-CEAbot',beam_size=5))

commands.append(run_dialog("v1.0",abot_startfrom=abot_huber_diverse,
                            qbot_startfrom='01-May-19-04:23:08-Wedv1.0_drop_0.5_SL-Qbot_0.1_cel_1000_ftl_ActualHuber_0.007finetune_job/qbot_ep_75.vd'
                           ,additional_job_spec='DivQbot-DivAbot',beam_size=5))

commands.append(run_dialog("v1.0",abot_startfrom=abot_huber_diverse,
                            qbot_startfrom=qBot_epoch,
                           additional_job_spec='CEQbot-DivAbot',beam_size=5))

commands.append(run_dialog("v1.0",abot_startfrom='13-Mar-19-01:34:58-Wedv1.0_drop_0_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_job/abot_ep_4.vd',
                            qbot_startfrom='13-Mar-19-01:34:58-Wedv1.0_drop_0_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_job/qbot_ep_4.vd',
                           additional_job_spec='ICCV-RLQbot-ICCV-RLAbot',beam_size=5))

commands.append(run_dialog("v1.0",abot_startfrom='15-May-19-15:44:49-WedRL-Bots_v1.0_drop_0_SLABotRL_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_disc_0.5Chkpt2_ActualHuberFinetune_0.07_job/abot_ep_8.vd',
                            qbot_startfrom='15-May-19-15:44:49-WedRL-Bots_v1.0_drop_0_SLABotRL_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_disc_0.5Chkpt2_ActualHuberFinetune_0.07_job/qbot_ep_8.vd',
                           additional_job_spec='RLQbot-RLAbot-Finetuned-CEAbot',beam_size=5))

commands.append(run_dialog("v1.0",abot_startfrom='13-May-19-06:14:15-MonRL-Bots_v1.0_drop_0_RL-Bots_40_cel_1000_ftl_100_rlc_anneal_3_1_disc_0.5ActualHuberFinetune_0.07_job/abot_ep_4.vd',
                            qbot_startfrom='13-May-19-06:14:15-MonRL-Bots_v1.0_drop_0_RL-Bots_40_cel_1000_ftl_100_rlc_anneal_3_1_disc_0.5ActualHuberFinetune_0.07_job/qbot_ep_4.vd',
                           additional_job_spec='RLQbot-RLAbot-Finetuned-DivAbot',beam_size=5))
'''
# commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
#                             qbot_startfrom=qBot_epoch,additional_job_spec='Test-Ent',beam_size=5))


# commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
#                             qbot_startfrom='01-May-19-04:23:08-Wedv1.0_drop_0.5_SL-Qbot_0.1_cel_1000_ftl_ActualHuber_0.007finetune_job/qbot_ep_75.vd',
#           i                    additional_job_spec='DivQbot-CEAbot',beam_size=1))

# commands.append(run_dialog("v1.0",abot_startfrom='06-Mar-19-05:40:09-Wedv1.0_drop_0_SL-Abot_1_cel_ActualHuber_0.001Finetune_job/abot_ep_10.vd',
#                             qbot_startfrom='01-May-19-04:23:08-Wedv1.0_drop_0.5_SL-Qbot_0.1_cel_1000_ftl_ActualHuber_0.007finetune_job/qbot_ep_75.vd'
#                            ,additional_job_spec='DivQbot-DivAbot',beam_size=1))

# commands.append(run_dialog("v1.0",abot_startfrom='06-Mar-19-05:40:09-Wedv1.0_drop_0_SL-Abot_1_cel_ActualHuber_0.001Finetune_job/abot_ep_10.vd',
#                             qbot_startfrom=qBot_epoch,
#                            additional_job_spec='CEQbot-DivAbot',beam_size=1))

'''
commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="26-Feb-19-08:07:02-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.1Finetune_ep78_job"
                                           "/qbot_ep_100.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_0.1_ep100"))


commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.1Finetune_job"
                                           "/qbot_ep_78.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_0.1_finetune_ep78"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.5job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_0.5_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.5Finetune_job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_0.5_finetune_ep2"))


commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_1job/"
                                           "qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_1.0_ep_2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_1Finetune_job/"
                                           "qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_1.0_fintune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_5job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_5_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_5Finetune_job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_5_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.1job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_0.1_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.1Finetune_job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_0.1_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_50job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_50_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_50Finetune_job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_50_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_100job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_100_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_100Finetune_job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_100_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_500job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_500_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_500Finetune_job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_500_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_1000job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_1000_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_1000Finetune_job/"
                                           "qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Coeff_1000_finetune_ep2"))

'''

'''
commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-15:21:52-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_10Finetune_FrozenDecoder_job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Frozen_Coeff_10_finetune_ep64"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-15:21:52-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_5Finetune_FrozenDecoder_job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Frozen_Coeff_5_finetune_ep64"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-15:21:52-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_2Finetune_FrozenDecoder_job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Frozen_Coeff_2_finetune_ep64"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-15:21:52-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_1Finetune_FrozenDecoder_job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Frozen_Coeff_1_finetune_ep64"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-15:21:52-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.5Finetune_FrozenDecoder_job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Frozen_Coeff_0.5_finetune_ep64"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-15:21:52-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.1Finetune_FrozenDecoder_job"
                                           "/qbot_ep_64.vd"
                            ,additional_job_spec="SLActualHuber_Frozen_Coeff_0.1_finetune_ep64"))

'''

'''
commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-15:21:52-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_10Finetune_FrozenDecoder_job"
                                           "/qbot_ep_2.vd"
                            ,additional_job_spec="SLCos_Frozen_Coeff_10_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-15:21:52-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_5Finetune_FrozenDecoder_job"
                                           "/qbot_ep_2.vd"
                            ,additional_job_spec="SLCos_Frozen_Coeff_5_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-15:21:52-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_2Finetune_FrozenDecoder_job"
                                           "/qbot_ep_2.vd"
                            ,additional_job_spec="SLCos_Frozen_Coeff_2_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-15:21:52-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_1Finetune_FrozenDecoder_job"
                                           "/qbot_ep_2.vd"
                            ,additional_job_spec="SLCos_Frozen_Coeff_1_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-15:21:52-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_0.5Finetune_FrozenDecoder_job"
                                           "/qbot_ep_2.vd"
                            ,additional_job_spec="SLCos_Frozen_Coeff_0.5_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-15:21:52-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_0.1Finetune_FrozenDecoder_job"
                                           "/qbot_ep_2.vd"
                            ,additional_job_spec="SLCos_Frozen_Coeff_0.1_finetune_ep2"))
'''


'''
commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-19:49:47-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_50ActualHuber_50Finetune_FrozenDecoder_job"
                                           "/qbot_ep_1.vd"
                            ,additional_job_spec="SLCosHuber_Frozen_Coeff_50_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-19:49:47-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_10ActualHuber_10Finetune_FrozenDecoder_job"
                                           "/qbot_ep_1.vd"
                            ,additional_job_spec="SLCosHuber_Frozen_Coeff_10_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-19:49:47-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_5ActualHuber_5Finetune_FrozenDecoder_job"
                                           "/qbot_ep_1.vd"
                            ,additional_job_spec="SLCosHuber_Frozen_Coeff_5_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-19:49:47-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_2ActualHuber_2Finetune_FrozenDecoder_job"
                                           "/qbot_ep_1.vd"
                            ,additional_job_spec="SLCosHuber_Frozen_Coeff_2_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-19:49:47-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_1ActualHuber_1Finetune_FrozenDecoder_job"
                                           "/qbot_ep_1.vd"
                            ,additional_job_spec="SLCosHuber_Frozen_Coeff_1_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-19:49:47-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_0.5ActualHuber_0.5Finetune_FrozenDecoder_job"
                                           "/qbot_ep_1.vd"
                            ,additional_job_spec="SLCosHuber_Frozen_Coeff_0.5_finetune_ep2"))

commands.append(run_dialog("v1.0",abot_startfrom=aBot_epoch,
                            qbot_startfrom="23-Feb-19-19:49:47-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_0.1ActualHuber_0.1Finetune_FrozenDecoder_job"
                                           "/qbot_ep_1.vd"
                            ,additional_job_spec="SLCosHuber_Frozen_Coeff_0.1_finetune_ep2"))

'''



# abot_checkpt = "abot_sl_ep60.vd"

# commands.append(run_dialog("v0.5",abot_startfrom=abot_checkpt,
#                            qbot_startfrom="30-Jan-19-17:41:13-Wed_6004707/qbot_ep_47.vd"
#                            ,additional_job_spec="Huber_5"))
# commands.append(run_dialog("v0.5",abot_startfrom=abot_checkpt,
#                            qbot_startfrom="30-Jan-19-05:43:49-Wed_1882439/qbot_ep_21.vd"
#                            ,additional_job_spec="SLBaseline"))
# commands.append(run_dialog("v0.5",abot_startfrom=abot_checkpt,
#                            qbot_startfrom="30-Jan-19-17:49:23-Wed_6702222/qbot_ep_36.vd",
#                            additional_job_spec="Cos_5"))

# commands.append(run_dialog("v0.5",abot_startfrom=abot_checkpt,
#                            qbot_startfrom="02-Feb-19-16:12:50-Sat_7816776/qbot_ep_45.vd",
#                            additional_job_spec="Huber5_Cos_5"))
'''
cmd = run_sl_rl_jobs("v0.5", "RL-Bots",
                     rl_abot_startfrom=abot_checkpt,cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Actor-Critic",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, rl_feat_loss_coeff=1000, rl_ce_loss_coeff=200, annealing_end=3, dropout=0,discount=0.9)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Actor-Critic",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=20000, rl_feat_loss_coeff=10000, rl_ce_loss_coeff=200, annealing_end=3, dropout=0,discount=0.9)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Actor-Critic",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, annealing_end=3, dropout=0,discount=0.9)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Actor-Critic",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, rl_feat_loss_coeff=10000, rl_ce_loss_coeff=100, annealing_end=3, dropout=0,discount=0.9)
commands.append(cmd)

                     rl_qbot_startfrom=os.path.join("30-Jan-19-17:49:23-Wed_6702222", "qbot_ep_36.vd"),
                     rl_loss_coeff=20000,
                     additional_job_spec="Cos_pretrain_5")
commands.append(cmd)
cmd = run_sl_rl_jobs("v0.5", "RL-Bots",
                     rl_abot_startfrom=abot_checkpt,
                     rl_qbot_startfrom="02-Feb-19-16:12:50-Sat_7816776/qbot_ep_45.vd",
                     rl_loss_coeff=20000,
                     additional_job_spec="Cos_Huber_pretrain_5")
commands.append(cmd)
'''

'''
cos_qbot_checkpt = "16-Feb-19-19:52:56-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_2job/qbot_ep_47.vd"

cmd =run_sl_rl_jobs("v1.0","RL-Bots",
    rl_abot_startfrom=aBot_epoch,
    rl_qbot_startfrom=cos_qbot_checkpt,
    rl_loss_coeff=20000,annealing_end=3,dropout=0,additional_job_spec="Cos")
commands.append(cmd)
'''
'''
Rl_dirs = ["26-Feb-19-19:46:33-Tuev1.0_drop_0_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_ActualHuber_Pretrain_ep100_EWA0.1_job",
           '26-Feb-19-19:46:33-Tuev1.0_drop_0_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_4_1_ActualHuber_Pretrain_ep100_EWA0.1_job',
           '26-Feb-19-19:46:33-Tuev1.0_drop_0_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_5_1_ActualHuber_Pretrain_ep100_EWA0.1_job',
           "26-Feb-19-19:46:33-Tuev1.0_drop_0_RL-Bots_1_cel_1000_ftl_1_rlc_anneal_3_1_ActualHuber_Pretrain_ep100_EWA0.1_job",
           "26-Feb-19-19:46:33-Tuev1.0_drop_0_RL-Bots_1_cel_1000_ftl_1_rlc_anneal_4_1_ActualHuber_Pretrain_ep100_EWA0.1_job",
           "26-Feb-19-19:46:33-Tuev1.0_drop_0_RL-Bots_1_cel_1000_ftl_1_rlc_anneal_5_1_ActualHuber_Pretrain_ep100_EWA0.1_job",
           "26-Feb-19-19:46:33-Tuev1.0_drop_0_RL-Bots_200_cel_10000_ftl_20000_rlc_anneal_3_1_ActualHuber_Pretrain_ep100_EWA0.1_job",
           "26-Feb-19-19:46:33-Tuev1.0_drop_0_RL-Bots_200_cel_10000_ftl_20000_rlc_anneal_4_1_ActualHuber_Pretrain_ep100_EWA0.1_job",
           '26-Feb-19-19:46:33-Tuev1.0_drop_0_RL-Bots_200_cel_10000_ftl_20000_rlc_anneal_5_1_ActualHuber_Pretrain_ep100_EWA0.1_job']

for ind in range(len(Rl_dirs)):
    cmd = run_val_plot_jobs("v1.0", 11, 15, eval_mode_list='ABotRank QABotsRank', abot_startfrom= Rl_dirs[ind],
        qbot_startfrom= Rl_dirs[ind], additional_job_spec=Rl_dirs[ind])
    commands.append(cmd)
'''

'''
Abot_diversity_dirs = \
    ["16-Feb-19-17:18:09-Satv1.0_drop_0.5_anneal_3_1_SL-Abot_1_cel_Huber_3.5continue_from_ep65_job",
     "16-Feb-19-17:18:09-Satv1.0_drop_0.5_anneal_3_1_SL-Abot_1_cel_Huber_4.5continue_from_ep65_job",
    "16-Feb-19-17:18:09-Satv1.0_drop_0.5_anneal_3_1_SL-Abot_1_cel_Huber_4continue_from_ep65_job"]

for ind in range(len(Abot_diversity_dirs)):
    cmd = run_val_plot_jobs("v1.0", 66, 90, eval_mode_list='ABotRank', abot_startfrom= Abot_diversity_dirs[ind],
                            additional_job_spec=Abot_diversity_dirs[ind])
    commands.append(cmd)
'''
'''
Rl_dirs = ['19-Feb-19-19:48:06-Tuev1.0_drop_0_RL-Bots_1_cel_1000_ftl_20000_rlc_anneal_3_1_Cos_job',
           '24-Feb-19-08:45:21-Sunv1.0_drop_0_RL-Bots_200_cel_10000_ftl_20000_rlc_anneal_3_1_Huber_Pretrain_ep504.5_job',
           '24-Feb-19-08:52:50-Sunv1.0_drop_0_RL-Bots_100_cel_10000_ftl_1_rlc_anneal_3_1_Huber_Pretrain_ep644.5_job'
           ]

for ind in range(len(Rl_dirs)):
    cmd = run_val_plot_jobs("v1.0", 1, 22, eval_mode_list='ABotRank', abot_startfrom= Rl_dirs[ind],
        qbot_startfrom= Rl_dirs[ind], additional_job_spec=Rl_dirs[ind] + "_NDCG")
    commands.append(cmd)
'''

# cmd = run_val_plot_jobs("v1.0", 1, 85, eval_mode_list='QBotRank', qbot_startfrom="09-Apr-19-02:35:47-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.01finetune_job",
#                         additional_job_spec="09-Apr-19-02:35:47-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.01finetune_job",use_actual_huber_loss=1, actual_huber_loss_coeff=0.01)
# commands.append(cmd)

# cmd = run_val_plot_jobs("v1.0", 1, 85, eval_mode_list='QBotRank', qbot_startfrom="23-Apr-19-16:28:50-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_finetune_with_initial_params_job",
#                         additional_job_spec="23-Apr-19-16:28:50-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_finetune_with_initial_params_job",use_actual_huber_loss=1, actual_huber_loss_coeff=1,
#                         use_cos_similarity_loss=1,cos_similarity_loss_coeff=1,use_huber_loss=1,huber_loss_coeff=1)
# commands.append(cmd)
#


# cmd = run_val_plot_jobs("v1.0", 1, 85, eval_mode_list='QBotRank', qbot_startfrom="09-Apr-19-02:35:47-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Huber_0.01finetune_job",
#                         additional_job_spec="09-Apr-19-02:35:47-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Huber_0.01finetune_job",use_huber_loss=1,huber_loss_coeff=0.01)
# commands.append(cmd)
#
# cmd = run_val_plot_jobs("v1.0", 1, 85, eval_mode_list='QBotRank', qbot_startfrom="23-Apr-19-04:01:43-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Huber_1finetune_job",
#                         additional_job_spec="23-Apr-19-04:01:43-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Huber_1finetune_job",use_huber_loss=1,huber_loss_coeff=1)
# commands.append(cmd)
#
# cmd = run_val_plot_jobs("v1.0", 1, 85, eval_mode_list='QBotRank', qbot_startfrom="23-Apr-19-04:01:43-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Huber_5finetune_job",
#                         additional_job_spec="23-Apr-19-04:01:43-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Huber_5finetune_job",use_huber_loss=1,huber_loss_coeff=5)
# commands.append(cmd)
#
# cmd = run_val_plot_jobs("v1.0", 1, 85, eval_mode_list='QBotRank', qbot_startfrom="23-Apr-19-04:01:43-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Huber_50finetune_job",
#                         additional_job_spec="23-Apr-19-04:01:43-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Huber_50finetune_job",use_huber_loss=1,huber_loss_coeff=50)
# commands.append(cmd)

'''
cmd = run_val_plot_jobs("v1.0", 1, 85, eval_mode_list='QBotRank', qbot_startfrom="09-Apr-19-02:35:47-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_0.01finetune_job",
                        additional_job_spec="09-Apr-19-02:35:47-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_0.01finetune_job",use_cos_similarity_loss=1,cos_similarity_loss_coeff=0.01)
commands.append(cmd)

cmd = run_val_plot_jobs("v1.0", 1, 85, eval_mode_list='QBotRank', qbot_startfrom="23-Apr-19-04:01:43-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_1finetune_job",
                        additional_job_spec="23-Apr-19-04:01:43-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_1finetune_job",use_cos_similarity_loss=1,cos_similarity_loss_coeff=1)
commands.append(cmd)

cmd = run_val_plot_jobs("v1.0", 1, 85, eval_mode_list='QBotRank', qbot_startfrom="23-Apr-19-04:01:43-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_5finetune_job",
                        additional_job_spec="23-Apr-19-04:01:43-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_5finetune_job",use_cos_similarity_loss=1,cos_similarity_loss_coeff=5)
commands.append(cmd)

cmd = run_val_plot_jobs("v1.0", 1, 85, eval_mode_list='QBotRank', qbot_startfrom="23-Apr-19-04:01:43-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_50finetune_job",
                        additional_job_spec="23-Apr-19-04:01:43-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_50finetune_job",use_cos_similarity_loss=1,cos_similarity_loss_coeff=50)
commands.append(cmd)
'''

# cmd = run_val_plot_jobs("v1.0", 1, 85, eval_mode_list='QBotRank', qbot_startfrom="01-Mar-19-06:10:45-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_50Finetune_job",
#                         additional_job_spec="01-Mar-19-06:10:45-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_50Finetune_job",use_actual_huber_loss=1, actual_huber_loss_coeff=50)
# commands.append(cmd)



# cmd = run_val_plot_jobs("v1.0", 1, 85, eval_mode_list='QBotRank', qbot_startfrom="09-Feb-19-06:29:48-Sat_7609112",
#                         additional_job_spec="09-Feb-19-06:29:48-Sat_7609112",use_actual_huber_loss=1,actual_huber_loss_coeff=1,
#                         use_huber_loss=1,huber_loss_coeff=1,use_cos_similarity_loss=1,cos_similarity_loss_coeff=1)
# commands.append(cmd)

'''
cos_checkpt = "16-Feb-19-19:52:56-Satv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_2job/qbot_ep_84.vd"
cos_loss_coeffs = [2]
cos_diverse_abot = '03-Apr-19-04:31:03-Wedv1.0_drop_0_SL-Abot_1_cel_Cos_0.1Finetune_job/abot_ep_9.vd'
for ind in range(len(cos_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=cos_diverse_abot,
        rl_qbot_startfrom=cos_checkpt,
        rl_loss_coeff=1,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="Cos_Pretrain_ep84_Diverse_CosAbot%s"%str(cos_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(cos_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=cos_diverse_abot,
        rl_qbot_startfrom=cos_checkpt,
        rl_loss_coeff=20000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=200,annealing_end=3,dropout=0,additional_job_spec="Cos_Pretrain_ep84_Diverse_CosAbot%s"%str(cos_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(cos_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=cos_diverse_abot,
        rl_qbot_startfrom=cos_checkpt,
        rl_loss_coeff=1,annealing_end=3,dropout=0,additional_job_spec="Cos_Pretrain_ep84_Diverse_CosAbot%s"%str(cos_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(cos_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=cos_diverse_abot,
        rl_qbot_startfrom=cos_checkpt,
        rl_loss_coeff=1,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=4,dropout=0,additional_job_spec="Cos_Pretrain_ep84_Diverse_CosAbot%s"%str(cos_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(cos_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=cos_diverse_abot,
        rl_qbot_startfrom=cos_checkpt,
        rl_loss_coeff=20000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=200,annealing_end=4,dropout=0,additional_job_spec="Cos_Pretrain_ep84_Diverse_CosAbot%s"%str(cos_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(cos_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=cos_diverse_abot,
        rl_qbot_startfrom=cos_checkpt,
        rl_loss_coeff=1,annealing_end=4,dropout=0,additional_job_spec="Cos_Pretrain_ep84_Diverse_CosAbot%s"%str(cos_loss_coeffs[ind])
                        )
    commands.append(cmd)
'''

# Huber RL

# huber_checkpt = "08-Feb-19-23:24:39-Fri_9391277/qbot_ep_64.vd"
# huber_loss_coeffs = [4.5]
# huber_diverse_abot = '04-Apr-19-16:09:01-Thuv1.0_drop_0_SL-Abot_1_cel_Huber_0.01Finetune_job/abot_ep_5.vd'

# for ind in range(len(huber_loss_coeffs)):
#     cmd =run_sl_rl_jobs("v1.0","RL-Bots",
#         rl_abot_startfrom=huber_diverse_abot,
#         rl_qbot_startfrom=huber_checkpt,
#         rl_loss_coeff=20000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=250,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
#                         )
#     commands.append(cmd)
#
# for ind in range(len(huber_loss_coeffs)):
#     cmd =run_sl_rl_jobs("v1.0","RL-Bots",
#         rl_abot_startfrom=huber_diverse_abot,
#         rl_qbot_startfrom=huber_checkpt,
#         rl_loss_coeff=20000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=300,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
#                         )
#     commands.append(cmd)
#
# for ind in range(len(huber_loss_coeffs)):
#     cmd =run_sl_rl_jobs("v1.0","RL-Bots",
#         rl_abot_startfrom=huber_diverse_abot,
#         rl_qbot_startfrom=huber_checkpt,
#         rl_loss_coeff=20000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=150,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
#                         )
#     commands.append(cmd)
#
# for ind in range(len(huber_loss_coeffs)):
#     cmd =run_sl_rl_jobs("v1.0","RL-Bots",
#         rl_abot_startfrom=huber_diverse_abot,
#         rl_qbot_startfrom=huber_checkpt,
#         rl_loss_coeff=20000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=350,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
#                         )
#     commands.append(cmd)
#
# for ind in range(len(huber_loss_coeffs)):
#     cmd =run_sl_rl_jobs("v1.0","RL-Bots",
#         rl_abot_startfrom=huber_diverse_abot,
#         rl_qbot_startfrom=huber_checkpt,
#         rl_loss_coeff=20000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
#                         )
#     commands.append(cmd)
#
# for ind in range(len(huber_loss_coeffs)):
#     cmd =run_sl_rl_jobs("v1.0","RL-Bots",
#         rl_abot_startfrom=huber_diverse_abot,
#         rl_qbot_startfrom=huber_checkpt,
#         rl_loss_coeff=20000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=400,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
#                         )
#     commands.append(cmd)
#

'''
for ind in range(len(huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_diverse_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=200,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=10,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_diverse_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=10,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
                        )
    commands.append(cmd)

# for ind in range(len(huber_loss_coeffs)):
#     cmd =run_sl_rl_jobs("v1.0","RL-Bots",
#         rl_abot_startfrom=huber_diverse_abot,
#         rl_qbot_startfrom=huber_checkpt,
#         rl_loss_coeff=10,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
#                         )
#     commands.append(cmd)

for ind in range(len(huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_diverse_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=20,rl_feat_loss_coeff=100,rl_ce_loss_coeff=2,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_diverse_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=5000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_diverse_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=12500,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=200,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_diverse_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=13000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=200,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
                        )
    commands.append(cmd)
    
'''

'''
for ind in range(len(huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_diverse_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=50000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=500,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_diverse_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=50000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=5000,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_diverse_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=40000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=1000,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_diverse_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=40000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=10000,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_diverse_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=2000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_diverse_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=200000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=200,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_diverse_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=200,rl_feat_loss_coeff=100000,rl_ce_loss_coeff=200,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_diverse_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=200000,rl_feat_loss_coeff=100000,rl_ce_loss_coeff=2000,annealing_end=3,dropout=0,additional_job_spec="Huber_Pretrain_ep64_Diverse_HuberAbot%s"%str(huber_loss_coeffs[ind])
                        )
    commands.append(cmd)
'''
# RL Baseline

# cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Future-Reward",
#                      rl_abot_startfrom=aBot_epoch,
#                      rl_qbot_startfrom=qBot_epoch,
#                      rl_loss_coeff=1, rl_feat_loss_coeff=1000, rl_ce_loss_coeff=200, annealing_end=3, dropout=0)
# commands.append(cmd)

'''
cmd = run_sl_rl_jobs("v1.0", "RL-Bots",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, rl_feat_loss_coeff=1000, rl_ce_loss_coeff=200, annealing_end=3, dropout=0,additional_job_spec="EWA")
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=20000, rl_feat_loss_coeff=10000, rl_ce_loss_coeff=200, annealing_end=3, dropout=0,additional_job_spec="EWA")
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, annealing_end=3, dropout=0,additional_job_spec="EWA")
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, rl_feat_loss_coeff=10000, rl_ce_loss_coeff=100, annealing_end=3, dropout=0,additional_job_spec="EWA")
commands.append(cmd)
'''
'''
# Huber + Abot
ActualHuberLossCoeffs = [0.1,1,2,5,10]
for hlc in ActualHuberLossCoeffs:
    commands.append(run_sl_rl_jobs("v1.0","SL-Abot",
                   use_actual_huber_loss=1,actual_huber_loss_coeff=hlc,dropout=0))

'''

#Actual Huber RL
'''actual_huber_checkpt = "09-Apr-19-02:35:47-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.001finetune_job/qbot_ep_60.vd"
actual_huber_div_abot = '06-Mar-19-05:40:09-Wedv1.0_drop_0_SL-Abot_1_cel_ActualHuber_0.001Finetune_job/abot_ep_10.vd'
actual_huber_loss_coeffs = [0.5]

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=actual_huber_div_abot,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=200,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=10,annealing_end=3,dropout=0,additional_job_spec="DivAHuber-Abot-Qbot")

    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=actual_huber_div_abot,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=10,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,additional_job_spec="DivAHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=actual_huber_div_abot,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=20,rl_feat_loss_coeff=100,rl_ce_loss_coeff=2,annealing_end=3,dropout=0,additional_job_spec="DivAHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=actual_huber_div_abot,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=100,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=30,annealing_end=3,dropout=0,additional_job_spec="DivAHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=actual_huber_div_abot,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=5,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="DivAHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=actual_huber_div_abot,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=10,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="DivAHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=actual_huber_div_abot,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=20,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="DivAHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=actual_huber_div_abot,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=100,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="DivAHuber-Abot-Qbot")
    commands.append(cmd)
'''
# Cos-RL

#Actual Huber RL
'''
cos_checkpt = "09-Apr-19-02:35:47-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Cos_0.01finetune_job/qbot_ep_64.vd"
cos_div_abot = "03-Apr-19-04:31:03-Wedv1.0_drop_0_SL-Abot_1_cel_Cos_0.1Finetune_job/abot_ep_9.vd"
actual_huber_loss_coeffs = [0.5]
for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=cos_div_abot,
        rl_qbot_startfrom=cos_checkpt,
        rl_loss_coeff=200,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=10,annealing_end=3,dropout=0,additional_job_spec="DivCos-Abot-Qbot")

    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=cos_div_abot,
        rl_qbot_startfrom=cos_checkpt,
        rl_loss_coeff=10,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,additional_job_spec="DivCos-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=cos_div_abot,
        rl_qbot_startfrom=cos_checkpt,
        rl_loss_coeff=20,rl_feat_loss_coeff=100,rl_ce_loss_coeff=2,annealing_end=3,dropout=0,additional_job_spec="DivCos-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=cos_div_abot,
        rl_qbot_startfrom=cos_checkpt,
        rl_loss_coeff=100,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=30,annealing_end=3,dropout=0,additional_job_spec="DivCos-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=cos_div_abot,
        rl_qbot_startfrom=cos_checkpt,
        rl_loss_coeff=5,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="DivCos-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=cos_div_abot,
        rl_qbot_startfrom=cos_checkpt,
        rl_loss_coeff=10,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="DivCos-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=cos_div_abot,
        rl_qbot_startfrom=cos_checkpt,
        rl_loss_coeff=20,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="DivCos-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=cos_div_abot,
        rl_qbot_startfrom=cos_checkpt,
        rl_loss_coeff=100,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="DivCos-Abot-Qbot")
    commands.append(cmd)
'''


'''
huber_checkpt = "09-Apr-19-02:35:47-Tuev1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_Huber_0.0001finetune_job/qbot_ep_84.vd"
huber_div_abot = "04-Apr-19-16:09:01-Thuv1.0_drop_0_SL-Abot_1_cel_Huber_0.01Finetune_job/abot_ep_5.vd"
actual_huber_loss_coeffs = [0.5]
for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_div_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=10,rl_feat_loss_coeff=1000,rl_ce_loss_coeff= 20,annealing_end=3,dropout=0,additional_job_spec="DivHuber-Abot-Qbot")

    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_div_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=20,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,additional_job_spec="DivHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_div_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=5,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=30,annealing_end=3,dropout=0,additional_job_spec="DivHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_div_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=3,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=50,annealing_end=3,dropout=0,additional_job_spec="DivHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_div_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=4,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="DivHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_div_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=6,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="DivHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_div_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=7,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="DivHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_div_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=2,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=5,annealing_end=3,dropout=0,additional_job_spec="DivHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_div_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=6,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=10,annealing_end=3,dropout=0,additional_job_spec="DivHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_div_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=5,rl_feat_loss_coeff=100,rl_ce_loss_coeff=10,annealing_end=3,dropout=0,additional_job_spec="DivHuber-Abot-Qbot")
    commands.append(cmd)


for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_div_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=50,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=5,annealing_end=3,dropout=0,additional_job_spec="DivHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_div_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=30,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=3,annealing_end=3,dropout=0,additional_job_spec="DivHuber-Abot-Qbot")
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=huber_div_abot,
        rl_qbot_startfrom=huber_checkpt,
        rl_loss_coeff=5,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=30,annealing_end=3,dropout=0,additional_job_spec="DivHuber-Abot-Qbot")
    commands.append(cmd)


'''


# for ind in range(len(actual_huber_loss_coeffs)):
#     cmd =run_sl_rl_jobs("v1.0","RL-Bots",
#         rl_abot_startfrom=huber_div_abot,
#         rl_qbot_startfrom=huber_checkpt,
#         rl_loss_coeff=100,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="DivHuber-Abot-Qbot")
#     commands.append(cmd)

#Actual Huber Qbot jobs
'''
Actual_Huber_Qbot_dirs = ['22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.1job',
           '22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_0.5job',
           '22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_1000job',
           '22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_100job',
           '22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_10job',
           '22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_1job',
           '22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_500job',
           '22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_50job',
           '22-Feb-19-07:51:58-Friv1.0_drop_0.5_SL-Qbot_1_cel_1000_ftl_ActualHuber_5job']

for ind in range(len(Actual_Huber_Qbot_dirs)):
    cmd = run_val_plot_jobs("v1.0", 1, 85, eval_mode_list='QBotRank dialog',
                            qbot_startfrom= Actual_Huber_Qbot_dirs[ind], additional_job_spec=Actual_Huber_Qbot_dirs[ind])
    commands.append(cmd)
'''

'''
actual_huber_checkpt = "01-May-19-04:23:08-Wedv1.0_drop_0.5_SL-Qbot_0.1_cel_1000_ftl_ActualHuber_0.007finetune_job/qbot_ep_75.vd"
actual_huber_loss_coeffs = [0.07]
# abot_huber_diverse = "09-May-19-14:58:47-Thuv1.0_drop_0_SL-Abot_1_cel_ActualHuber_5e-05Finetune_job/abot_ep_6.vd"
abot_huber_diverse = aBot_epoch


for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=20,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=4,dropout=0,use_ndcg=True,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=10,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=4,dropout=0,use_ndcg=True,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=2,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=4,dropout=0,use_ndcg=True,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)


for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=20,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=2,dropout=0,use_ndcg=True,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=10,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=2,dropout=0,use_ndcg=True,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=2,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=2,dropout=0,use_ndcg=True,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)
'''

'''
for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=30,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=2,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=15,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=5,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=10,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=20,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=100,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)


for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=10,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=100,rl_feat_loss_coeff=100,rl_ce_loss_coeff=10,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=30,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=3,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)
'''



'''
for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=20000,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=200,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=1,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=100,rl_feat_loss_coeff=1000,rl_ce_loss_coeff= 200,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))

    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=200,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=50,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=abot_huber_diverse,
        rl_loss_coeff=20,rl_feat_loss_coeff=5000,rl_ce_loss_coeff=200,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=50,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=1000,rl_feat_loss_coeff=5000,rl_ce_loss_coeff=2000,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)



for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=50,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=5,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=5,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=5,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=5,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)


for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=50,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=100,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

'''

'''
for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=110,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=40,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=90,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=40,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=80,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=40,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=110,rl_feat_loss_coeff=100,rl_ce_loss_coeff=10,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=90,rl_feat_loss_coeff=100,rl_ce_loss_coeff=10,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=120,rl_feat_loss_coeff=100,rl_ce_loss_coeff=10,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)


for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=240,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=180,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=220,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

#########################################

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=110,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=50,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=90,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=30,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=80,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=50,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=110,rl_feat_loss_coeff=100,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=90,rl_feat_loss_coeff=100,rl_ce_loss_coeff=15,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=120,rl_feat_loss_coeff=100,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)


for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=240,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=30,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=180,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=15,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=220,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=30,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

'''

'''

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=100,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=10,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=100,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=10,annealing_end=3,dropout=0,additional_job_spec="Chkpt2_ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=80,rl_feat_loss_coeff=100,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=40,rl_feat_loss_coeff=100,rl_ce_loss_coeff=10,annealing_end=3,dropout=0,use_ndcg=True,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind])
                        )
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=30,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=30,annealing_end=3,dropout=0,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))

    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=50,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=25,annealing_end=3,dropout=0,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=abot_huber_diverse,
        rl_loss_coeff=50,rl_feat_loss_coeff=5000,rl_ce_loss_coeff=25,annealing_end=3,dropout=0,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=100,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=30,annealing_end=3,dropout=0,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=30,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=30,annealing_end=3,dropout=0,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=100,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=40,annealing_end=3,dropout=0,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=100,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=25,annealing_end=3,dropout=0,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=200,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=100,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=5,annealing_end=3,dropout=0,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=20,rl_feat_loss_coeff=100,rl_ce_loss_coeff=20,annealing_end=3,dropout=0,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=50,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=10,annealing_end=3,dropout=0,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=10,rl_feat_loss_coeff=1000,rl_ce_loss_coeff=50,annealing_end=3,dropout=0,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

for ind in range(len(actual_huber_loss_coeffs)):
    cmd =run_sl_rl_jobs("v1.0","RL-Bots",
        rl_abot_startfrom=abot_huber_diverse,
        rl_qbot_startfrom=actual_huber_checkpt,
        rl_loss_coeff=20,rl_feat_loss_coeff=10000,rl_ce_loss_coeff=40,annealing_end=3,dropout=0,additional_job_spec="ActualHuberFinetune_%s"%str(actual_huber_loss_coeffs[ind]))
    commands.append(cmd)

'''
'''
cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Multi-GPU",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=100000, rl_feat_loss_coeff=1000, rl_ce_loss_coeff=0.1, annealing_end=3, dropout=0, batchMultiply=25)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Multi-GPU",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, rl_feat_loss_coeff=10000, rl_ce_loss_coeff=100, annealing_end=3, dropout=0, batchMultiply=25)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Multi-GPU",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=20000, rl_feat_loss_coeff=10000, rl_ce_loss_coeff=200, annealing_end=3, dropout=0, batchMultiply=25)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Multi-GPU",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, annealing_end=3, dropout=0, batchMultiply=25)
commands.append(cmd)



cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Multi-GPU",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=100, rl_feat_loss_coeff=1000, rl_ce_loss_coeff=100, annealing_end=3, dropout=0, batchMultiply=25)
commands.append(cmd)
cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Multi-GPU",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=100000, rl_feat_loss_coeff=1000, rl_ce_loss_coeff=0.1, annealing_end=3, dropout=0, batchMultiply=10)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Multi-GPU",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, rl_feat_loss_coeff=10000, rl_ce_loss_coeff=100, annealing_end=3, dropout=0, batchMultiply=10)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Multi-GPU",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=20000, rl_feat_loss_coeff=10000, rl_ce_loss_coeff=200, annealing_end=3, dropout=0, batchMultiply=10)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Multi-GPU",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, annealing_end=3, dropout=0, batchMultiply=10)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Multi-GPU",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=100, rl_feat_loss_coeff=1000, rl_ce_loss_coeff=100, annealing_end=3, dropout=0, batchMultiply=10)
commands.append(cmd)
'''
'''
cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Actor-Critic",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, rl_feat_loss_coeff=1000, rl_ce_loss_coeff=200, annealing_end=3, dropout=0,discount=0.9)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Actor-Critic",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=20000, rl_feat_loss_coeff=10000, rl_ce_loss_coeff=200, annealing_end=3, dropout=0,discount=0.9)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Actor-Critic",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, annealing_end=3, dropout=0,discount=0.9)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Actor-Critic",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, rl_feat_loss_coeff=10000, rl_ce_loss_coeff=100, annealing_end=3, dropout=0,discount=0.9)
commands.append(cmd)


cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Actor-Critic",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, rl_feat_loss_coeff=1000, rl_ce_loss_coeff=200, annealing_end=3, dropout=0,discount=0.99)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Actor-Critic",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=20000, rl_feat_loss_coeff=10000, rl_ce_loss_coeff=200, annealing_end=3, dropout=0,discount=0.99)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Actor-Critic",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, annealing_end=3, dropout=0,discount=0.99)
commands.append(cmd)

cmd = run_sl_rl_jobs("v1.0", "RL-Bots-Actor-Critic",
                     rl_abot_startfrom=aBot_epoch,
                     rl_qbot_startfrom=qBot_epoch,
                     rl_loss_coeff=1, rl_feat_loss_coeff=10000, rl_ce_loss_coeff=100, annealing_end=3, dropout=0,discount=0.99)
commands.append(cmd)
'''

# commands.append(run_sl_rl_jobs("v1.0","SL-Qbot"
#                ,rl_qbot_startfrom=qBot_epoch,additional_job_spec="finetune_with_initial_params"))

# HuberLossCoeffs = [0.01,0.03,0.05,0.07,0.1,0.2]
# for hlc in HuberLossCoeffs:
#     commands.append(run_sl_rl_jobs("v1.0","SL-Qbot",
#                    rl_ce_loss_coeff=0.07,use_huber_loss=1,huber_loss_coeff=hlc,rl_qbot_startfrom=qBot_epoch,additional_job_spec="finetune"))

# HuberLossCoeffs = [0.07,0.09,0.12]
# for hlc in HuberLossCoeffs:
#     commands.append(run_sl_rl_jobs("v1.0","SL-Qbot",
#                    rl_ce_loss_coeff=0.1,use_huber_loss=1,huber_loss_coeff=hlc,rl_qbot_startfrom=qBot_epoch,additional_job_spec="finetune"))

# HuberLossCoeffs = [0.005,0.01,0.05,0.1]
# for hlc in HuberLossCoeffs:
#     commands.append(run_sl_rl_jobs("v1.0","SL-Qbot",
#                    rl_ce_loss_coeff=0.01,use_huber_loss=1,huber_loss_coeff=hlc,rl_qbot_startfrom=qBot_epoch,additional_job_spec="finetune"))

# CosLossCoeffs = [0.1,0.05,0.01,0.005,0.001]
# for clc in CosLossCoeffs:
#     commands.append(run_sl_rl_jobs("v1.0","SL-Qbot",rl_ce_loss_coeff=0.1,
#                    use_cos_similarity_loss=1,cos_similarity_loss_coeff=clc,rl_qbot_startfrom=qBot_epoch,additional_job_spec="finetune"))

# ActualHuberLossCoeffs = [0.006,0.008]
#
# for ahlc in ActualHuberLossCoeffs:
#     commands.append(run_sl_rl_jobs("v1.0","SL-Qbot-Multi-Round",
#                    rl_ce_loss_coeff=0.1,use_actual_huber_loss=1,actual_huber_loss_coeff=ahlc,rl_qbot_startfrom=qBot_epoch,additional_job_spec="finetune"))

# ActualHuberLossCoeffs = [0.0002,0.0003,0.0006,0.0008,0.001,0.002,0.003]
#
# for ahlc in ActualHuberLossCoeffs:
#     commands.append(run_sl_rl_jobs("v1.0","SL-Qbot-Multi-Round",
#                    rl_ce_loss_coeff=0.05,use_actual_huber_loss=1,actual_huber_loss_coeff=ahlc,rl_qbot_startfrom=qBot_epoch,additional_job_spec="finetune"))

run_commands(commands)
