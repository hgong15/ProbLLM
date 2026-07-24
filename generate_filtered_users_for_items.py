# Llama-SubTower Training & Evaluation
# -----------------------------------
# 生成冷启动物品的评估样本，输出格式为概率（与训练样本一致）
#

import os
import json
import pickle
import torch
from torch import nn
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import torch.nn.functional as F


class LlamaHead(nn.Module):
    """Llama-SubTower模型头，用于学习用户-物品表示"""
    def __init__(self, input_size, hidden_size, output_size):
        super(LlamaHead, self).__init__()
        self.user_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, user_origin_emb, item_origin_emb):
        user_content_emb = self.user_mlp(user_origin_emb)
        item_content_emb = self.item_mlp(item_origin_emb)
        sim = torch.matmul(user_content_emb, item_content_emb.t())
        logits = torch.sigmoid(torch.diag(sim))
        return logits

    def calculate_score(self, user_origin_emb, item_origin_emb):
        user_content_emb = self.user_mlp(user_origin_emb)
        item_content_emb = self.item_mlp(item_origin_emb)
        sim = torch.matmul(user_content_emb, item_content_emb.t())
        return sim


def is_movie_dataset(dataset: str) -> bool:
    return dataset in {"Movielens", "MovieLens", "ml-1m", "ml-10m", "ml-10m_1"}


def main(args):
    # 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_item_targets(csv_name, entity_type):
        path = os.path.join(dataset_root, csv_name)
        if not os.path.exists(path):
            return {}
        df = pd.read_csv(path)
        if len(df) == 0:
            return {}
        grouped = df.groupby("item")["user"].agg(list).to_dict()
        return {int(item): entity_type for item in grouped.keys()}

    # 数据集路径
    dataset_root = os.path.join("data", args.dataset)
    # 加载 strict cold-start 和 warm-up 物品。warm-up 物品已有少量 support 交互，但仍需要模拟补充交互。
    target_item_types = {}
    for csv_name in ("cold_item_val.csv", "cold_item_test.csv"):
        target_item_types.update(load_item_targets(csv_name, "strict_cold"))
    for csv_name in ("warmup_val.csv", "warmup_test.csv"):
        target_item_types.update(load_item_targets(csv_name, "warmup"))
    all_target_items = sorted(target_item_types.keys())
    if not all_target_items:
        raise ValueError("No strict-cold or warm-up target items found for candidate generation.")

    # 物品原始数据路径
    item_content_path = os.path.join(dataset_root, "raw-data.csv")

    # 加载LLM嵌入
    item_origin_emb = torch.load(os.path.join(dataset_root, args.item_emb_file), map_location="cpu").float().to(device)
    user_origin_emb = torch.load(os.path.join(dataset_root, args.user_emb_file), map_location="cpu").float().to(device)
    if item_origin_emb.shape[1] != user_origin_emb.shape[1]:
        raise ValueError(
            f"Item/user LLM embedding dims differ: {item_origin_emb.shape[1]} vs {user_origin_emb.shape[1]}"
        )
    input_size = item_origin_emb.shape[1]
    if args.input_size is not None and args.input_size != input_size:
        print(f"[WARN] --input_size={args.input_size} does not match embedding dim={input_size}; using {input_size}.")

    # 初始化模型
    llama_model = LlamaHead(input_size, args.hidden_size, args.output_size).to(device)

    # 加载训练好的LlamaHead模型权重
    llama_model.load_state_dict(torch.load(os.path.join(dataset_root, args.llama_head_file), map_location="cpu"))
    if args.normalize_filter:
        with torch.no_grad():
            user_content_emb = F.normalize(llama_model.user_mlp(user_origin_emb), p=2, dim=1)
            item_content_emb = F.normalize(llama_model.item_mlp(item_origin_emb), p=2, dim=1)
            llama_all_score = torch.matmul(user_content_emb, item_content_emb.t())
        print("Filtering score: cosine over normalized LLM mapped embeddings.")
    else:
        llama_all_score = llama_model.calculate_score(user_origin_emb, item_origin_emb)
        print("Filtering score: raw dot product over LLM mapped embeddings.")

    if args.filter_source == "llm":
        all_score = llama_all_score[:, : len(item_origin_emb)].t()
        print("Filtering source: LLM mapped embeddings only.")
    else:
        # Some reproduced operating points use precomputed auxiliary embeddings
        # for candidate retrieval. This repository consumes those tensors when
        # supplied, but does not include the external code used to create them.
        ALDI_user_emb = torch.load(os.path.join(dataset_root, "ALDI_user_emb.pt"), map_location="cpu").to(device)
        ALDI_item_emb = torch.load(os.path.join(dataset_root, "ALDI_item_emb.pt"), map_location="cpu").to(device)
        ALDI_score = torch.matmul(ALDI_user_emb, ALDI_item_emb.t())
        if args.filter_source == "aldi":
            all_score = ALDI_score.t()
            print("Filtering source: ALDI embeddings only.")
        else:
            all_score = llama_all_score[:, : len(item_origin_emb)].t() + ALDI_score.t()
            print("Filtering source: hybrid LLM mapped embeddings + ALDI embeddings.")
    observed_train_users_by_item = {}
    train_path = os.path.join(dataset_root, "warm_emb.csv")
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        observed_train_users_by_item = train_df.groupby("item")["user"].agg(set).to_dict()

    def select_candidate_users(item_id):
        observed_users = observed_train_users_by_item.get(item_id, set())
        candidate_pool_size = min(
            all_score.shape[1],
            max(args.topk * 5, args.topk + len(observed_users) + 20),
        )
        _, ranked_users = torch.topk(all_score[item_id], candidate_pool_size)
        selected_users = []
        for user in ranked_users.tolist():
            user_id = int(user)
            if user_id in observed_users:
                continue
            selected_users.append(user_id)
            if len(selected_users) >= args.topk:
                break
        return selected_users

    # 加载用户偏好和物品内容
    if args.dataset == "CiteULike":
        item_content = pd.read_csv(item_content_path, encoding="latin1")
    else:
        item_content = pd.read_csv(item_content_path)
    user_pref_path = os.path.join(dataset_root, "train_user_preference_list.pkl")
    with open(user_pref_path, "rb") as f:
        train_user_preference_list = pickle.load(f)


    # 生成用户-物品对和评估指令数据
    user_item_pair = []
    llama_instruction = []

    print("生成用户-物品对和评估指令数据...")

    if args.dataset == 'CiteULike':
        # 1. 修改指令模板：输出概率（与训练样本一致）
        instruction_template = "Given the user's interaction paper set, predict the probability (a value between 0 and 1, e.g., 0.85) that the user will like the target paper."
        for target_item in tqdm(all_target_items, desc="处理论文物品"):
            user_list = select_candidate_users(target_item)
            entity_type = target_item_types[target_item]
            for user_id in user_list:
                # 2. 修改输入文本格式：与训练样本的"User preference: ..."对齐
                instruction_input = f'User preference: "{train_user_preference_list[user_id]}", What is the probability the user will like the target paper "{item_content.iloc[target_item].title}"?'
                # 3. 修改输出：用0.5作为临时概率（后续替换为模型推理结果），并保留user_id和item_id
                temp_dict = {
                    "instruction": instruction_template,
                    "input": instruction_input,
                    "output": "1",  # 临时占位符
                    "user_id": user_id,  # 便于后续匹配用户
                    "item_id": target_item,  # 便于后续匹配物品
                    "entity_type": entity_type,
                }
                user_item_pair.append([user_id, target_item, entity_type])
                llama_instruction.append(temp_dict)
    elif is_movie_dataset(args.dataset):
        # 1. 修改指令模板：输出概率
        instruction_template = "Given the user's interaction movie set, predict the probability (a value between 0 and 1, e.g., 0.32) that the user will like the target movie."
        for target_item in tqdm(all_target_items, desc="处理电影物品"):
            user_list = select_candidate_users(target_item)
            entity_type = target_item_types[target_item]
            for user_id in user_list:
                # 2. 修改输入文本格式
                instruction_input = f'User preference: "{train_user_preference_list[user_id]}", What is the probability the user will like the target movie "{item_content.iloc[target_item].title}"?'
                # 3. 修改输出：临时概率+保留ID
                temp_dict = {
                    "instruction": instruction_template,
                    "input": instruction_input,
                    "output": "1",
                    "user_id": user_id,
                    "item_id": target_item,
                    "entity_type": entity_type,
                }
                user_item_pair.append([user_id, target_item, entity_type])
                llama_instruction.append(temp_dict)

    # 保存用户-物品对
    save_path = args.top20_output or os.path.join(dataset_root, "top20.csv")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    pd.DataFrame(user_item_pair, columns=["user", "item", "entity_type"]).to_csv(save_path, index=False)
    print(f"用户-物品对已保存至 {save_path}")

    # 保存评估指令数据（概率格式）
    instruction_path = args.eval_output or os.path.join(dataset_root, f"{args.dataset}_eval.json")
    os.makedirs(os.path.dirname(os.path.abspath(instruction_path)), exist_ok=True)
    with open(instruction_path, "w", encoding="utf-8") as f:
        json.dump(llama_instruction, f, indent=4, ensure_ascii=False)
    print(f"评估指令数据已保存至 {instruction_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成冷启动物品的评估样本（概率输出格式）")
    # 模型参数
    parser.add_argument("--input_size", type=int, default=None, help="LLM embedding dim (inferred by default)")
    parser.add_argument("--hidden_size", type=int, default=1024, help="MLP隐藏层维度")
    parser.add_argument("--output_size", type=int, default=200, help="输出嵌入维度")
    parser.add_argument("--item_emb_file", type=str, default="llm_item_content_emb.pt")
    parser.add_argument("--user_emb_file", type=str, default="llm_user_content_emb.pt")
    parser.add_argument("--llama_head_file", type=str, default="llama_head.bin")
    # 数据集和评估参数
    parser.add_argument("--dataset", type=str, default="CiteULike", help="数据集名称")
    parser.add_argument("--topk", type=int, default=20, help="每个物品选取的Top-K用户数")
    parser.add_argument(
        "--filter_source",
        choices=["llm", "aldi", "hybrid"],
        default="llm",
        help="Candidate filter source. hybrid combines LLM mapped scores with precomputed auxiliary embeddings.",
    )
    parser.add_argument(
        "--normalize_filter",
        action="store_true",
        help="Use L2-normalized LLM mapped embeddings for candidate filtering.",
    )
    parser.add_argument("--top20_output", type=str, default=None)
    parser.add_argument("--eval_output", type=str, default=None)
    args = parser.parse_args()
    main(args)
