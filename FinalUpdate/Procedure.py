'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import os
import world
import numpy as np
import torch
import utils
from utils import timer
import model
import multiprocessing

CORES = multiprocessing.cpu_count() // 2
_SCORE_PRIOR_CACHE = None
_GROUP_OFFSET_CACHE = None


def combine_masks(*arrays):
    valid_arrays = [np.asarray(array, dtype=np.int64) for array in arrays if array is not None and len(array) > 0]
    if not valid_arrays:
        return None
    return np.unique(np.concatenate(valid_arrays, axis=0))


def use_group_candidate_eval():
    mode = os.environ.get('PROBLLM_EVAL_CANDIDATE_MODE', 'full').strip().lower()
    return mode in {'group', 'grouped', 'split', 'target'}


def score_prior_matrix(dataset):
    global _SCORE_PRIOR_CACHE
    prior_file = os.environ.get('PROBLLM_SCORE_PRIOR_FILE', '').strip()
    alpha = float(os.environ.get('PROBLLM_SCORE_PRIOR_ALPHA', '0') or 0)
    if not prior_file or alpha == 0:
        return None
    cache_key = (prior_file, alpha, dataset.n_users, dataset.m_items)
    if _SCORE_PRIOR_CACHE is not None and _SCORE_PRIOR_CACHE[0] == cache_key:
        return _SCORE_PRIOR_CACHE[1]
    import csv
    prob_column = os.environ.get('PROBLLM_SCORE_PRIOR_COLUMN', 'probability')
    prior = torch.zeros((dataset.n_users, dataset.m_items), dtype=torch.float32, device=world.device)
    n_rows = 0
    with open(prior_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            user = int(row.get('user', row.get('uid')))
            item = int(row.get('item', row.get('iid')))
            if user < 0 or user >= dataset.n_users or item < 0 or item >= dataset.m_items:
                continue
            prob = float(row.get(prob_column, row.get('prob', 1.0)) or 1.0)
            prob = min(max(prob, 0.0), 1.0)
            value = alpha * prob
            if value > float(prior[user, item].item()):
                prior[user, item] = value
            n_rows += 1
    print(f"[SCORE-PRIOR] file={prior_file} alpha={alpha} rows={n_rows}")
    _SCORE_PRIOR_CACHE = (cache_key, prior)
    return prior


def group_score_offset_vector(dataset):
    global _GROUP_OFFSET_CACHE
    strict_offset = float(os.environ.get('PROBLLM_GROUP_OFFSET_STRICT', '0') or 0)
    warmup_offset = float(os.environ.get('PROBLLM_GROUP_OFFSET_WARMUP', '0') or 0)
    warm_offset = float(os.environ.get('PROBLLM_GROUP_OFFSET_WARM', '0') or 0)
    if strict_offset == 0 and warmup_offset == 0 and warm_offset == 0:
        return None
    cache_key = (strict_offset, warmup_offset, warm_offset, dataset.m_items)
    if _GROUP_OFFSET_CACHE is not None and _GROUP_OFFSET_CACHE[0] == cache_key:
        return _GROUP_OFFSET_CACHE[1]
    warm_item, strict_cold_item, warmup_item = dataset.eval_item_groups()
    offset = torch.zeros((dataset.m_items,), dtype=torch.float32, device=world.device)
    if len(strict_cold_item) > 0 and strict_offset != 0:
        offset[torch.as_tensor(strict_cold_item, dtype=torch.long, device=world.device)] = strict_offset
    if len(warmup_item) > 0 and warmup_offset != 0:
        offset[torch.as_tensor(warmup_item, dtype=torch.long, device=world.device)] = warmup_offset
    if len(warm_item) > 0 and warm_offset != 0:
        offset[torch.as_tensor(warm_item, dtype=torch.long, device=world.device)] = warm_offset
    print(
        "[GROUP-OFFSET] "
        f"strict={strict_offset} warmup={warmup_offset} warm={warm_offset} "
        f"items(strict/warmup/warm)={len(strict_cold_item)}/{len(warmup_item)}/{len(warm_item)}"
    )
    _GROUP_OFFSET_CACHE = (cache_key, offset)
    return offset


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class
    
    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()
    if world.config.get('rwft_weighted', False):
        posWeights = torch.Tensor(dataset.getTrainPairWeights(S[:, 0], S[:, 1])).float()
    else:
        posWeights = None

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    if posWeights is not None:
        posWeights = posWeights.to(world.device)
        users, posItems, negItems, posWeights = utils.shuffle(users, posItems, negItems, posWeights)
    else:
        users, posItems, negItems = utils.shuffle(users, posItems, negItems)
    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.
    if posWeights is not None:
        batches = utils.minibatch(users, posItems, negItems, posWeights, batch_size=world.config['bpr_batch_size'])
    else:
        batches = utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])
    for batch_i, batch in enumerate(batches):
        if posWeights is not None:
            batch_users, batch_pos, batch_neg, batch_weight = batch
        else:
            batch_users, batch_pos, batch_neg = batch
            batch_weight = None
        cri = bpr.stageOne(batch_users, batch_pos, batch_neg, pos_weight=batch_weight)
        aver_loss += cri
        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)
    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss{aver_loss:.3f}-{time_info}"
    
    
def test_one_batch(X):
    sorted_items = X[0]
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}
        
            
def Test(dataset, Recmodel, mode='test'):
    warm_item, strict_cold_item, warmup_item = dataset.eval_item_groups()
    cold_object = getattr(dataset, 'para_dict', {}).get('cold_object', 'item')
    if cold_object == 'user':
        # User-side cold start evaluates cold/warm-up/warm users while still
        # recommending train-visible items. Mask items unseen in the mixed
        # training graph instead of applying item-side cold/warm group masks.
        visible_item_mask = combine_masks(dataset.para_dict.get('mixed_cold_item', np.array([], dtype=np.int32)))
        strict_cold_mask = visible_item_mask
        warmup_mask = visible_item_mask
        warm_mask = visible_item_mask
    elif not use_group_candidate_eval():
        strict_cold_mask = None
        warmup_mask = None
        warm_mask = None
    else:
        strict_cold_mask = combine_masks(warm_item, warmup_item)
        warmup_mask = combine_masks(warm_item, strict_cold_item)
        warm_mask = combine_masks(strict_cold_item, warmup_item)

    if mode == 'test':
        cold_user_nb, warmup_user_nb, warm_user_nb, overall_user_nb = dataset.test_user_nb()
        cold_user, warmup_user, warm_user, overall_user = dataset.test_user()
        exclude_cold, exclude_warmup, exclude_warm, exclude_overall = dataset.test_exclude()
        cold_res, _ = test(dataset, Recmodel, cold_user_nb, cold_user, exclude_cold, masked_items=strict_cold_mask)
        warmup_res, _ = test(dataset, Recmodel, warmup_user_nb, warmup_user, exclude_warmup, masked_items=warmup_mask)
        warm_res, _ = test(dataset, Recmodel, warm_user_nb, warm_user, exclude_warm, masked_items=warm_mask)
        overall_res, _ = test(dataset, Recmodel, overall_user_nb, overall_user, exclude_overall, masked_items=None)
    elif mode == 'val':
        cold_user_nb, warmup_user_nb, warm_user_nb, overall_user_nb = dataset.val_user_nb()
        cold_user, warmup_user, warm_user, overall_user = dataset.val_user()
        exclude_cold, exclude_warmup, exclude_warm, exclude_overall = dataset.val_exclude()
        cold_res, _ = test(dataset, Recmodel, cold_user_nb, cold_user, exclude_cold, masked_items=strict_cold_mask)
        warmup_res, _ = test(dataset, Recmodel, warmup_user_nb, warmup_user, exclude_warmup, masked_items=warmup_mask)
        warm_res, _ = test(dataset, Recmodel, warm_user_nb, warm_user, exclude_warm, masked_items=warm_mask)
        overall_res, _ = test(dataset, Recmodel, overall_user_nb, overall_user, exclude_overall, masked_items=None)
    else:
        Exception("mode error")
    print("Strict Cold-Start Result:", cold_res)
    print("Warm-Up Result:", warmup_res)
    print("Warm Result:", warm_res)
    print("Overall Result:", overall_res)

    return cold_res, warmup_res, warm_res, overall_res


def test(dataset, Recmodel, ts_nei, ts_user, exclude_pair_cnt, masked_items=None):
    results = {'precision': np.zeros(len(world.topks)),
               'recall': np.zeros(len(world.topks)),
               'ndcg': np.zeros(len(world.topks))}
    if len(ts_user) == 0:
        return results, np.empty((0, max(world.topks)))

    max_K = max(world.topks)
    rating_list = []
    score_list = []
    groundTrue_list = []
    batch_size = world.config['test_u_batch_size']
    masked_items_tensor = None
    score_prior = score_prior_matrix(dataset)
    group_offset = group_score_offset_vector(dataset)
    if masked_items is not None and len(masked_items) > 0:
        masked_items_tensor = torch.as_tensor(masked_items, dtype=torch.long, device=world.device)
    for i, beg in enumerate(range(0, len(ts_user), batch_size)):
        end = min(beg + batch_size, len(ts_user))
        batch_user = ts_user[beg:end]
        groundTrue = ts_nei[batch_user]
        batch_user = torch.Tensor(batch_user).long().to(world.device)
        with torch.no_grad():
            rating_all_item = Recmodel.getUsersRating(batch_user)
            if score_prior is not None:
                rating_all_item = rating_all_item + score_prior[batch_user]
            if group_offset is not None:
                rating_all_item = rating_all_item + group_offset

        # ================== exclude =======================
        exclude_pair = exclude_pair_cnt[0][exclude_pair_cnt[1][i]:exclude_pair_cnt[1][i + 1]]
        if len(exclude_pair) > 0:
            exclude_rows = torch.as_tensor(exclude_pair[:, 0], dtype=torch.long, device=world.device)
            exclude_cols = torch.as_tensor(exclude_pair[:, 1], dtype=torch.long, device=world.device)
            rating_all_item[exclude_rows, exclude_cols] = -1e10

        if masked_items_tensor is not None:
            rating_all_item[:, masked_items_tensor] = -1e10
        # ===================================================

        top_scores, top_item_index = torch.topk(rating_all_item, k=max_K, dim=-1)

        score_list.append(top_scores.detach().cpu().numpy())
        rating_list.append(top_item_index.detach().cpu().numpy())
        groundTrue_list.append(groundTrue)

    X = zip(rating_list, groundTrue_list)
    pre_results = list(map(test_one_batch, X))
    for result in pre_results:
        results['recall'] += result['recall']
        results['precision'] += result['precision']
        results['ndcg'] += result['ndcg']
    n_ts_user = float(len(ts_user))
    results['recall'] /= n_ts_user
    results['precision'] /= n_ts_user
    results['ndcg'] /= n_ts_user
    return results, np.concatenate(score_list, axis=0)

def topk_numpy(arr, k, dim):
    idx = np.argpartition(-arr,kth=k,axis=dim)
    idx = idx.take(indices=range(k),axis=dim)
    val = np.take_along_axis(arr,indices=idx,axis=dim)
    sorted_idx = np.argsort(-val,axis=dim)
    idx = np.take_along_axis(idx,indices=sorted_idx,axis=dim)
    val = np.take_along_axis(val,indices=sorted_idx,axis=dim)
    return val,idx

def get_top_k(ratings, k):
    topk_val, topk_idx = topk_numpy(ratings, k, dim=-1)
    return topk_val, topk_idx
