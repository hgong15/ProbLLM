'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''

import os
from os.path import join
import torch
from enum import Enum
from parse import parse_args
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
args = parse_args()

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
CODE_PATH = join(ROOT_PATH, 'code')
DATA_PATH = join(ROOT_PATH, 'data')
BOARD_PATH = join(CODE_PATH, 'runs')
FILE_PATH = join(CODE_PATH, 'checkpoints')
import sys
sys.path.append(join(CODE_PATH, 'sources'))


if not os.path.exists(FILE_PATH):
    os.makedirs(FILE_PATH, exist_ok=True)


config = {}
all_dataset = ['LastMF', 'gowalla', 'yelp2018', 'amazon-book','XING', 'CiteULike','ml-1m', 'amazon-beauty','ml-10m', 'ml-10m_1', 'book-crossing']
all_models  = ['mf', 'lgn']
# config['batch_size'] = 4096
config['bpr_batch_size'] = args.bpr_batch
config['latent_dim_rec'] = args.recdim
config['lightGCN_n_layers']= args.layer
config['dropout'] = args.dropout
config['keep_prob']  = args.keepprob
config['A_n_fold'] = args.a_fold
config['test_u_batch_size'] = args.testbatch
config['multicore'] = args.multicore
config['lr'] = args.lr
config['decay'] = args.decay
config['pretrain'] = args.pretrain
config['A_split'] = False
config['bigdata'] = False
config['file_name'] = args.file_name
config['extended_file'] = args.extended_file
config['graph_cache'] = args.graph_cache
config['rwft_weighted'] = bool(args.rwft_weighted)
config['rwft_beta'] = args.rwft_beta
config['caga_gamma'] = args.caga_gamma
config['caga_k0'] = args.caga_k0
config['caga_target_object'] = args.caga_target_object
config['sim_prob_column'] = args.sim_prob_column
config['target_only_update'] = os.environ.get('FINALUPDATE_TARGET_ONLY_UPDATE', '0') == '1'
config['target_only_object'] = os.environ.get('FINALUPDATE_TARGET_ONLY_OBJECT', 'user')
config['target_only_freeze_items'] = os.environ.get('FINALUPDATE_TARGET_ONLY_FREEZE_ITEMS', '1') != '0'
config['target_only_freeze_users'] = os.environ.get('FINALUPDATE_TARGET_ONLY_FREEZE_USERS', '1') != '0'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
GPU = torch.cuda.is_available() and os.environ.get('FINALUPDATE_FORCE_CPU', '0') != '1'
device = torch.device('cuda' if GPU else "cpu")
#device = torch.device('1' if GPU else "cpu")
#torch.cuda.set_device('1')
CORES = multiprocessing.cpu_count() // 2
seed = args.seed

dataset = args.dataset
model_name = args.model
is_book_crossing_alias = dataset.startswith('book-crossing_')
is_amazon23_alias = dataset.startswith('amazon23_')
if dataset not in all_dataset and not is_book_crossing_alias and not is_amazon23_alias:
    raise NotImplementedError(f"Haven't supported {dataset} yet!, try {all_dataset}")
if model_name not in all_models:
    raise NotImplementedError(f"Haven't supported {model_name} yet!, try {all_models}")




TRAIN_epochs = args.epochs
LOAD = args.load
PATH = args.path
topks = eval(args.topks)
tensorboard = args.tensorboard
comment = args.comment
# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



def cprint(words : str):
    print(f"\033[0;30;43m{words}\033[0m")

logo = r"""
██╗      ██████╗ ███╗   ██╗
██║     ██╔════╝ ████╗  ██║
██║     ██║  ███╗██╔██╗ ██║
██║     ██║   ██║██║╚██╗██║
███████╗╚██████╔╝██║ ╚████║
╚══════╝ ╚═════╝ ╚═╝  ╚═══╝
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
# print(logo)
