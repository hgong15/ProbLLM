import os
import argparse
import pandas as pd
import numpy as np
import pickle
import json
from tqdm import tqdm
import random

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def gen_user_preference_list(dataset_dir: str, item_raw_data_filename: str) -> list:
    """
    根据用户的交互数据和物品信息生成每个用户的偏好列表。
    """
    train_file_name = 'warm_emb.csv'
    data_root = os.path.join('data', dataset_dir)
    train_file_path = os.path.join(data_root, train_file_name)


    train_file = pd.read_csv(train_file_path)
    train_file_groupby_user = train_file.groupby('user')
    n_user_item_path = os.path.join(data_root, 'n_user_item.pkl')
    with open(n_user_item_path, 'rb') as f:
        n_user_item = pickle.load(f)
    user_num = int(n_user_item['user'])

    train_file_groupby_user_list = []
    for user_id in range(user_num):
        if user_id in train_file_groupby_user.groups:
            train_file_groupby_user_list.append(list(train_file_groupby_user.get_group(user_id).item))
        else:
            train_file_groupby_user_list.append([])

    # 读取物品信息
    content_data_path = os.path.join(data_root, item_raw_data_filename + '.csv')
    if dataset_dir == 'CiteULike':
        content_data = pd.read_csv(content_data_path, encoding='latin1')
    else:
        content_data = pd.read_csv(content_data_path)

    # 构建用户偏好列表
    train_file_groupby_user_content_list = []
    train_file_groupby_user_content_target_list = []

    print("Processing user interactions...")
    for interaction_item_list in tqdm(train_file_groupby_user_list):
        if not interaction_item_list:
            train_file_groupby_user_content_list.append(['default info'])
            train_file_groupby_user_content_target_list.append('default info')
            continue
        interaction_item_list_len = min(len(interaction_item_list), 20)
        interaction_item_list = interaction_item_list[:interaction_item_list_len]
        target_item = interaction_item_list[-1]
        interaction_item_list = interaction_item_list[:-1]

        title_list = [content_data.iloc[int(id)].title for id in interaction_item_list]
        target_item_title = content_data.iloc[target_item].title
        train_file_groupby_user_content_list.append(title_list)
        train_file_groupby_user_content_target_list.append(target_item_title)

    train_user_preference_list = ['","'.join(title_list) for title_list in train_file_groupby_user_content_list]

    # 保存用户偏好列表
    preference_file = os.path.join(data_root, 'train_user_preference_list.pkl')
    with open(preference_file, 'wb') as f:
        pickle.dump(train_user_preference_list, f)
    print(f"Saved user preference list to {preference_file}")
    return [train_user_preference_list, train_file, content_data]

def is_movie_dataset(dataset_dir: str) -> bool:
    return dataset_dir in {"Movielens", "MovieLens", "ml-1m", "ml-10m", "ml-10m_1", "amazon23_movies_tv_item"}


def is_book_dataset(dataset_dir: str) -> bool:
    return dataset_dir in {"amazon23_books_item"}


def is_product_dataset(dataset_dir: str) -> bool:
    return dataset_dir.startswith("amazon23_") and dataset_dir.endswith("_item_y2023_rel50")


def generate_llama_json(dataset_dir: str, train_file: pd.DataFrame, item_raw_data: pd.DataFrame, train_user_preference_list: list):
    """
    根据用户交互数据生成 LLaMA SFT 样本 JSON 文件（输出概率）。
    """
    # 读取用户和物品总数
    dataset_root = os.path.join('data', dataset_dir)
    n_user_item_dict_path = os.path.join(dataset_root, 'n_user_item.pkl')
    with open(n_user_item_dict_path, 'rb') as f:
        num = pickle.load(f)
    item_num = num['item']

    all_interaction = []

    group_df = train_file.groupby('user')
    print("Constructing positive and negative samples...")
    for user, group in tqdm(group_df):
        # 正样本
        for item in group['item'].values:
            all_interaction.append([user, item, 1])

        # 负样本
        set_all = set(range(item_num))
        set_neg = set_all - set(group['item'].values)
        neg_size = min(len(group['item'].values), len(set_neg))
        if neg_size > 0:
            neg_samples = np.random.choice(sorted(set_neg), size=neg_size, replace=False)
            for neg_item in neg_samples:
                all_interaction.append([user, neg_item, 0])

    # 构造 LLaMA instruction JSON（输出概率）
    llama_instruction = []
    if dataset_dir == 'CiteULike':
        for user, item, action in tqdm(all_interaction):
            # 关键：将 NumPy int64 转为 Python int
            user = int(user)  # 转换user_id类型
            item = int(item)  # 转换item_id类型
            # 指令：要求输出0-1之间的概率
            instruction = "Given the user's interaction paper set, predict the probability (a value between 0 and 1, e.g., 0.85) that the user will like the target paper."
            # 正样本概率整体偏高（0.5-1.0），负样本整体偏低（0.0-0.5）
            if action == 1:
                prob = round(random.uniform(0.6, 1.0), 2)  # 保留2位小数
            else:
                prob = round(random.uniform(0.0, 0.4), 2)
            output = str(prob)  # 转为字符串格式（LLM训练样本通常用字符串）
            # 输入：用户偏好 + 目标物品
            instruction_input = f'User preference: "{train_user_preference_list[user]}", What is the probability the user will like the target paper "{item_raw_data.iloc[item]["title"]}"?'
            llama_instruction.append({
                "instruction": instruction,
                "input": instruction_input,
                "output": output,
                "user_id": user,  # 直接记录生成时的user索引
                "item_id": item   # 直接记录生成时的item索引
            })
    elif is_movie_dataset(dataset_dir):
        for user, item, action in tqdm(all_interaction):
            user = int(user)
            item = int(item)
            instruction = "Given the user's interaction movie set, predict the probability (a value between 0 and 1, e.g., 0.32) that the user will like the target movie."
            if action == 1:
                prob = round(random.uniform(0.6, 1.0), 2)
            else:
                prob = round(random.uniform(0.0, 0.4), 2)
            output = str(prob)
            instruction_input = f'User preference: "{train_user_preference_list[user]}", What is the probability the user will like the target movie "{item_raw_data.iloc[item]["title"]}"?'
            llama_instruction.append({
                "instruction": instruction,
                "input": instruction_input,
                "output": output,
                "user_id": user,  # 直接记录生成时的user索引
                "item_id": item   # 直接记录生成时的item索引
            })
    elif is_book_dataset(dataset_dir):
        for user, item, action in tqdm(all_interaction):
            user = int(user)
            item = int(item)
            instruction = "Given the user's interaction book set, predict the probability (a value between 0 and 1, e.g., 0.32) that the user will like the target book."
            if action == 1:
                prob = round(random.uniform(0.6, 1.0), 2)
            else:
                prob = round(random.uniform(0.0, 0.4), 2)
            output = str(prob)
            instruction_input = f'User preference: "{train_user_preference_list[user]}", What is the probability the user will like the target book "{item_raw_data.iloc[item]["title"]}"?'
            llama_instruction.append({
                "instruction": instruction,
                "input": instruction_input,
                "output": output,
                "user_id": user,
                "item_id": item
            })
    elif is_product_dataset(dataset_dir):
        for user, item, action in tqdm(all_interaction):
            user = int(user)
            item = int(item)
            instruction = "Given the user's interaction product set, predict the probability (a value between 0 and 1, e.g., 0.32) that the user will like the target product."
            if action == 1:
                prob = round(random.uniform(0.6, 1.0), 2)
            else:
                prob = round(random.uniform(0.0, 0.4), 2)
            output = str(prob)
            instruction_input = f'User preference: "{train_user_preference_list[user]}", What is the probability the user will like the target product "{item_raw_data.iloc[item]["title"]}"?'
            llama_instruction.append({
                "instruction": instruction,
                "input": instruction_input,
                "output": output,
                "user_id": user,
                "item_id": item
            })
    else:
        raise ValueError('Can\'t support this dataset.')

    random.shuffle(llama_instruction)
    json_path = os.path.join(dataset_root, 'train_sample.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(llama_instruction, f, indent=4, ensure_ascii=False)
    print(f"LLaMA training JSON (probability output) saved to {json_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate LLaMA SFT JSON from user-item interactions (output probability).")
    parser.add_argument('--dataset', type=str, default='CiteULike', help="Dataset root directory")
    parser.add_argument('--content_file', type=str, default='raw-data', help="Item raw data CSV file")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    train_user_preference_list, train_file, item_raw_data = gen_user_preference_list(args.dataset, args.content_file)
    generate_llama_json(args.dataset, train_file, item_raw_data, train_user_preference_list)

if __name__ == '__main__':
    main()
