'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--bpr_batch', type=int,default=4096,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=200,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=2,
                        help="the layer num of lightGCN")
    parser.add_argument('--lr', type=float,default=0.001,
                        help="the learning rate")
    parser.add_argument('--decay', type=float,default=1e-4,
                        help="the weight decay for l2 normalizaton")
    parser.add_argument('--dropout', type=int,default=0,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int, default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int, default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str, default='CiteULike',
                        help="available datasets: [lastfm, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--path', type=str, default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?', default="[20]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=int,default=0,
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=int,default=1)
    parser.add_argument('--epochs', type=int,default=100)
    parser.add_argument('--multicore', type=int, default=0, help='whether we use multiprocessing or not in test')
    parser.add_argument('--pretrain', type=int, default=0, help='whether we use pretrained weight or not')
    parser.add_argument('--seed', type=int, default=2020, help='random seed')
    parser.add_argument('--model', type=str, default='lgn', help='rec-model, support [mf, lgn]')
    parser.add_argument('--file_name', type=str, default="updated_model", help='file name')
    parser.add_argument(
        '--extended_file',
        type=str,
        default='predicted_cold_item_interaction.csv',
        help='CSV file of simulated user-item interactions, relative to the dataset directory unless absolute.',
    )
    parser.add_argument(
        '--graph_cache',
        type=str,
        default=None,
        help='Optional adjacency cache filename for this run. Defaults to fin_s_pre_adj_mat.npz for the standard extended file.',
    )
    parser.add_argument(
        '--rwft_weighted',
        type=int,
        default=0,
        help='Use reliability-weighted simulated interactions in BPR and graph construction.',
    )
    parser.add_argument(
        '--rwft_beta',
        type=float,
        default=1.0,
        help='Reliability coefficient for simulated interactions: weight = beta * probability.',
    )
    parser.add_argument(
        '--caga_gamma',
        type=float,
        default=0.0,
        help='Cold-aware decay rate for simulated edges. 0 disables the decay.',
    )
    parser.add_argument(
        '--caga_k0',
        type=int,
        default=5,
        help='Warm-up threshold used by the cold-aware simulated-edge decay.',
    )
    parser.add_argument(
        '--caga_target_object',
        type=str,
        choices=['user', 'item'],
        default='item',
        help='Target entity whose pre-augmentation degree controls CAGA.',
    )
    parser.add_argument(
        '--sim_prob_column',
        type=str,
        default='probability',
        help='Probability column in the simulated interaction CSV.',
    )
    return parser.parse_args()
