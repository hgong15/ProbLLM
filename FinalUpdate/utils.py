'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import world
import torch
from torch import optim
import numpy as np
from dataloader import BasicDataset
from time import time
from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import os
import csv
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    world.cprint("Cpp extension not loaded")
    sample_ext = False


class BPRLoss:
    def __init__(self,
                 recmodel : PairWiseModel,
                 config : dict,
                 dataset: BasicDataset = None):
        self.model = recmodel
        self.weight_decay = config['decay']
        self.lr = config['lr']
        self.opt = optim.Adam(recmodel.parameters(), lr=self.lr)
        self.target_only_update = bool(config.get('target_only_update', False))
        self.freeze_items = bool(config.get('target_only_freeze_items', True))
        self.freeze_users = bool(config.get('target_only_freeze_users', True))
        self.target_object = config.get('target_only_object', 'user')
        self.target_mask = None
        if self.target_only_update:
            if dataset is None:
                raise ValueError("dataset is required when target_only_update is enabled")
            para = getattr(dataset, 'para_dict', {})
            if self.target_object == 'user':
                strict = para.get('strict_cold_user', para.get('cold_user', np.array([], dtype=np.int64)))
                warmup = para.get('warmup_user', np.array([], dtype=np.int64))
                target_users = np.asarray(sorted(set(np.asarray(strict, dtype=np.int64).tolist()) | set(np.asarray(warmup, dtype=np.int64).tolist())), dtype=np.int64)
                if len(target_users) == 0:
                    raise ValueError("No strict/warm-up target users found for target-only update")
                mask = torch.zeros(recmodel.num_users, dtype=torch.bool, device=world.device)
                mask[torch.as_tensor(target_users, dtype=torch.long, device=world.device)] = True
                self.target_mask = mask
                print(
                    "[TARGET-ONLY-UPDATE] "
                    f"object=user target_users={len(target_users)} freeze_items={self.freeze_items}"
                )
            elif self.target_object == 'item':
                strict = para.get('strict_cold_item', para.get('cold_item', np.array([], dtype=np.int64)))
                warmup = para.get('warmup_item', np.array([], dtype=np.int64))
                target_items = np.asarray(sorted(set(np.asarray(strict, dtype=np.int64).tolist()) | set(np.asarray(warmup, dtype=np.int64).tolist())), dtype=np.int64)
                if len(target_items) == 0:
                    raise ValueError("No strict/warm-up target items found for target-only update")
                mask = torch.zeros(recmodel.num_items, dtype=torch.bool, device=world.device)
                mask[torch.as_tensor(target_items, dtype=torch.long, device=world.device)] = True
                self.target_mask = mask
                print(
                    "[TARGET-ONLY-UPDATE] "
                    f"object=item target_items={len(target_items)} freeze_users={self.freeze_users}"
                )
            else:
                raise ValueError(f"target_only_object must be 'user' or 'item', got {self.target_object!r}")

    def stageOne(self, users, pos, neg, pos_weight=None):
        loss, reg_loss = self.model.bpr_loss(users, pos, neg, pos_weight=pos_weight)
        reg_loss = reg_loss*self.weight_decay
        loss = loss + reg_loss

        self.opt.zero_grad()
        loss.backward()
        if self.target_only_update:
            with torch.no_grad():
                user_grad = getattr(self.model, 'embedding_user').weight.grad
                if self.target_object == 'user' and user_grad is not None:
                    user_grad[~self.target_mask] = 0
                item_grad = getattr(self.model, 'embedding_item').weight.grad
                if self.target_object == 'user' and self.freeze_items and item_grad is not None:
                    item_grad.zero_()
                if self.target_object == 'item' and item_grad is not None:
                    item_grad[~self.target_mask] = 0
                if self.target_object == 'item' and self.freeze_users and user_grad is not None:
                    user_grad.zero_()
        self.opt.step()

        return loss.cpu().item()


def UniformSample_original(dataset, neg_ratio = 1):
    dataset : BasicDataset
    allPos = dataset.allPos
    start = time()
    group_mode = os.environ.get("PROBLLM_NEGATIVE_GROUP_MODE", "").strip().lower()
    group_modes = {"same_eval_group", "same_group", "group", "hard_same_eval_group", "prior_hard", "hard_group"}
    if sample_ext and group_mode not in group_modes:
        S = sampling.sample_negative(dataset.n_users, dataset.m_items,
                                     dataset.trainDataSize, allPos, neg_ratio)
    else:
        S = UniformSample_original_python(dataset)
    return S

def _as_int_array(value):
    if isinstance(value, np.ndarray):
        return value.astype(np.int64, copy=False).reshape(-1)
    if isinstance(value, (list, tuple, set)):
        return np.asarray(list(value), dtype=np.int64).reshape(-1)
    return np.asarray([int(value)], dtype=np.int64)

def _eval_group_negative_cache(dataset):
    cache = getattr(dataset, "_probllm_eval_group_negative_cache", None)
    if cache is not None:
        return cache
    para = getattr(dataset, "para_dict", {})
    group = np.zeros(dataset.m_items, dtype=np.int8)
    warm_items = _as_int_array(para.get("warm_item", np.array([], dtype=np.int64)))
    strict_items = _as_int_array(para.get("strict_cold_item", para.get("cold_item", np.array([], dtype=np.int64))))
    warmup_items = _as_int_array(para.get("warmup_item", np.array([], dtype=np.int64)))
    group[warm_items] = 3
    group[strict_items] = 1
    group[warmup_items] = 2
    candidates = {
        gid: np.where(group == gid)[0].astype(np.int64)
        for gid in (0, 1, 2, 3)
    }
    cache = (group, candidates)
    setattr(dataset, "_probllm_eval_group_negative_cache", cache)
    print(
        "[NEGATIVE-GROUP-MODE] same_eval_group "
        f"items(strict/warmup/warm/default)="
        f"{len(candidates[1])}/{len(candidates[2])}/{len(candidates[3])}/{len(candidates[0])}"
    )
    return cache

def _sample_same_group_negative(dataset, positem, posForUser):
    group, candidates = _eval_group_negative_cache(dataset)
    gid = int(group[int(positem)])
    pool = candidates.get(gid)
    if pool is None or len(pool) <= 1:
        return None
    for _ in range(100):
        negitem = int(pool[np.random.randint(0, len(pool))])
        if negitem not in posForUser:
            return negitem
    return None

def _hard_negative_cache(dataset):
    cache = getattr(dataset, "_probllm_hard_negative_cache", None)
    prior_file = os.environ.get("PROBLLM_HARD_NEGATIVE_FILE", "").strip()
    if cache is not None and cache.get("prior_file") == prior_file:
        return cache
    if not prior_file:
        cache = {"prior_file": "", "user_group_pools": {}}
        setattr(dataset, "_probllm_hard_negative_cache", cache)
        return cache

    group, _candidates = _eval_group_negative_cache(dataset)
    user_group_lists = {}
    rows = 0
    kept = 0
    with open(prior_file, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            user = int(row.get("user", row.get("uid")))
            item = int(row.get("item", row.get("iid")))
            if user < 0 or user >= dataset.n_users or item < 0 or item >= dataset.m_items:
                continue
            gid = int(group[item])
            if gid <= 0:
                continue
            user_group_lists.setdefault(user, {}).setdefault(gid, []).append(item)
            rows += 1

    rng = np.random.default_rng(int(os.environ.get("PROBLLM_HARD_NEGATIVE_SEED", str(world.seed))))
    user_group_pools = {}
    for user, by_group in user_group_lists.items():
        user_pools = {}
        for gid, items in by_group.items():
            # Keep first occurrence order from the candidate file, but remove duplicates.
            seen = set()
            unique = []
            for item in items:
                if item not in seen:
                    seen.add(item)
                    unique.append(item)
            if os.environ.get("PROBLLM_HARD_NEGATIVE_SHUFFLE", "0") == "1":
                rng.shuffle(unique)
            if unique:
                user_pools[gid] = np.asarray(unique, dtype=np.int64)
                kept += len(unique)
        if user_pools:
            user_group_pools[user] = user_pools

    cache = {"prior_file": prior_file, "user_group_pools": user_group_pools}
    setattr(dataset, "_probllm_hard_negative_cache", cache)
    print(
        "[NEGATIVE-GROUP-MODE] hard_same_eval_group "
        f"file={prior_file} rows={rows} kept_unique={kept} users={len(user_group_pools)}"
    )
    return cache

def _sample_hard_same_group_negative(dataset, user, positem, posForUser):
    group, _candidates = _eval_group_negative_cache(dataset)
    gid = int(group[int(positem)])
    cache = _hard_negative_cache(dataset)
    pool = cache.get("user_group_pools", {}).get(int(user), {}).get(gid)
    if pool is None or len(pool) == 0:
        return None
    max_tries = int(os.environ.get("PROBLLM_HARD_NEGATIVE_MAX_TRIES", "100"))
    head_k = int(os.environ.get("PROBLLM_HARD_NEGATIVE_HEAD_K", "0") or 0)
    if head_k > 0:
        pool = pool[: min(head_k, len(pool))]
    for _ in range(max_tries):
        negitem = int(pool[np.random.randint(0, len(pool))])
        if negitem not in posForUser:
            return negitem
    return None

def UniformSample_original_python(dataset):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    total_start = time()
    dataset : BasicDataset
    user_num = dataset.trainDataSize
    users = np.random.randint(0, dataset.n_users, user_num)
    allPos = dataset.allPos
    group_mode = os.environ.get("PROBLLM_NEGATIVE_GROUP_MODE", "").strip().lower()
    same_group_neg = group_mode in {"same_eval_group", "same_group", "group"}
    hard_same_group_neg = group_mode in {"hard_same_eval_group", "prior_hard", "hard_group"}
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[user]
        if len(posForUser) == 0:
            continue
        sample_time2 += time() - start
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        negitem = None
        if hard_same_group_neg:
            negitem = _sample_hard_same_group_negative(dataset, user, positem, posForUser)
        if same_group_neg:
            negitem = _sample_same_group_negative(dataset, positem, posForUser)
        while True:
            if negitem is None:
                negitem = np.random.randint(0, dataset.m_items)
            if negitem in posForUser:
                negitem = None
                continue
            else:
                break
        S.append([user, positem, negitem])
        end = time()
        sample_time1 += end - start
    total = time() - total_start
    return np.array(S)

# ===================end samplers==========================
# =====================utils====================================

def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    if world.model_name == 'mf':
        file = f"mf-{world.config['file_name']}.pth.tar"
    elif world.model_name == 'lgn':
        file = f"lgn-{world.config['file_name']}.pth.tar"
    return os.path.join(world.FILE_PATH, file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result


class timer:
    """
    Time context manager for code block
        with timer():
            do something
        timer.get()
    """
    from time import time
    TAPE = [-1]  # global time record
    NAMED_TAPE = {}

    @staticmethod
    def get():
        if len(timer.TAPE) > 1:
            return timer.TAPE.pop()
        else:
            return -1

    @staticmethod
    def dict(select_keys=None):
        hint = "|"
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                hint = hint + f"{key}:{value:.2f}|"
        else:
            for key in select_keys:
                value = timer.NAMED_TAPE[key]
                hint = hint + f"{key}:{value:.2f}|"
        return hint

    @staticmethod
    def zero(select_keys=None):
        if select_keys is None:
            for key, value in timer.NAMED_TAPE.items():
                timer.NAMED_TAPE[key] = 0
        else:
            for key in select_keys:
                timer.NAMED_TAPE[key] = 0

    def __init__(self, tape=None, **kwargs):
        if kwargs.get('name'):
            timer.NAMED_TAPE[kwargs['name']] = timer.NAMED_TAPE[
                kwargs['name']] if timer.NAMED_TAPE.get(kwargs['name']) else 0.
            self.named = kwargs['name']
            if kwargs.get("group"):
                #TODO: add group function
                pass
        else:
            self.named = False
            self.tape = tape or timer.TAPE

    def __enter__(self):
        self.start = timer.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.named:
            timer.NAMED_TAPE[self.named] += timer.time() - self.start
        else:
            self.tape.append(timer.time() - self.start)


# ====================Metrics==============================
# =========================================================
def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}


def MRRatK_r(r, k):
    """
    Mean Reciprocal Rank
    """
    pred_data = r[:, :k]
    scores = np.log2(1./np.arange(1, k+1))
    pred_data = pred_data/scores
    pred_data = pred_data.sum(1)
    return np.sum(pred_data)

def NDCGatK_r(test_data,r,k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    """
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def AUC(all_item_scores, dataset, test_data):
    """
        design for a single user
    """
    dataset : BasicDataset
    r_all = np.zeros((dataset.m_items, ))
    r_all[test_data] = 1
    r = r_all[all_item_scores >= 0]
    test_item_scores = all_item_scores[all_item_scores >= 0]
    return roc_auc_score(r, test_item_scores)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

# ====================end Metrics=============================
# =========================================================
