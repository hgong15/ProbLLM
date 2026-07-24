# =====================================
# LLaMA SubTower Training Script (Soft Label Version)
# =====================================
import os
import random
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import get_scheduler
from torch.optim import AdamW
import argparse
import json


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================
# 1. 自定义Soft BCE损失函数（适配软标签）
# =====================================
class SoftBCEWithLogitsLoss(nn.Module):
    """适配软标签（0-1概率）的BCE损失，对应公式：L = -[y * log(σ(x)) + (1-y) * log(1-σ(x))]"""
    def __init__(self, weight=None, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        # inputs: 模型输出的原始分数（未过sigmoid）
        # targets: LLM输出的软标签概率（0-1连续值）
        loss = nn.BCEWithLogitsLoss(weight=self.weight, reduction='none')(inputs, targets)
        return loss.mean() if self.reduction == 'mean' else loss.sum()


# =====================================
# 2. 模型定义（输出原始分数，不过sigmoid）
# =====================================
class LlamaHead(nn.Module):
    """Two-tower MLP for user/item embeddings and interaction prediction"""
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
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
        sim = torch.matmul(user_content_emb, item_content_emb.t())  # 点积相似度
        return torch.diag(sim)  # 输出原始分数（未过sigmoid）


# =====================================
# 3. 数据集定义（支持软标签）
# =====================================
class InteractionDataset(Dataset):
    def __init__(self, user_ids, item_ids, labels):
        self.user_ids = torch.as_tensor(user_ids, dtype=torch.long)
        self.item_ids = torch.as_tensor(item_ids, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_ids[idx], self.item_ids[idx], self.labels[idx]


# =====================================
# 4. 训练循环
# =====================================
def train_loop(
    dataloader,
    model,
    loss_fn,
    optimizer,
    lr_scheduler,
    epoch,
    total_loss,
    device,
    user_emb,
    item_emb,
):
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    model.train()
    for step, (user_ids, item_ids, y) in enumerate(progress_bar, start=1):
        user_ids = user_ids.to(device, non_blocking=True)
        item_ids = item_ids.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        X1 = user_emb[user_ids]
        X2 = item_emb[item_ids]
        pred = model(X1, X2).float()  # 原始分数
        loss = loss_fn(pred, y)       # 软标签损失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        total_loss += loss.item()
        avg_loss = total_loss / ((epoch - 1) * len(dataloader) + step)
        progress_bar.set_postfix(loss=avg_loss)
    return total_loss


# =====================================
# 5. 评估循环
# =====================================
def test_loop(dataloader, model, device, user_emb, item_emb, mode="Test"):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for user_ids, item_ids, y in dataloader:
            user_ids = user_ids.to(device, non_blocking=True)
            item_ids = item_ids.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            X1 = user_emb[user_ids]
            X2 = item_emb[item_ids]
            pred_logits = model(X1, X2).float()
            pred_prob = torch.sigmoid(pred_logits)  # 转为概率
            pred = torch.where(pred_prob > 0.5, 1.0, 0.0)  # 硬标签预测
            y_hard = torch.where(y > 0.5, 1.0, 0.0)  # 软标签转硬标签
            total += y_hard.size(0)
            correct += (pred == y_hard).sum().item()
    acc = correct / total
    print(f"{mode} Accuracy: {100 * acc:.2f}%")
    return acc


# =====================================
# 6. 解析LLM输出的概率（直接读取user_id和item_id）
# =====================================
def load_llm_probs(json_path):
    """
    直接从JSON样本中读取user_id、item_id和概率：
    无需标题匹配，完全复用生成时的原始索引
    """
    llm_probs = {}
    with open(json_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)
    
    for idx, sample in enumerate(tqdm(samples, desc="Parsing LLM samples")):
        # 直接读取JSON中的user_id和item_id（生成时已确保是Python int）
        try:
            user_id = sample["user_id"]
            item_id = sample["item_id"]
            # 转换概率为浮点数（处理可能的字符串格式）
            prob = float(sample["output"].strip())
        except KeyError as e:
            print(f"警告：第{idx}个样本缺少字段「{e}」→ 跳过")
            continue
        except ValueError:
            print(f"警告：第{idx}个样本概率格式错误（{sample['output']}）→ 跳过")
            continue

        # 存储（user_id, item_id）→ 概率的映射
        llm_probs[(user_id, item_id)] = prob

    print(f"成功解析 {len(llm_probs)}/{len(samples)} 个样本")
    return llm_probs


# =====================================
# 7. 样本构建（使用LLM软标签）
# =====================================
def build_interaction_samples(train_interaction_df, item_num, llm_probs):
    user_ids = []
    item_ids = []
    labels = []
    group_df = train_interaction_df.groupby("user")
    print("[INFO] Building samples with LLM soft labels...")
    for user_id, group in tqdm(group_df):
        # 正样本：用户真实交互过的物品
        for item_id in group["item"].values:
            # 优先用LLM概率，无匹配时默认正样本概率1.0
            prob = llm_probs.get((user_id, item_id), 1.0)
            user_ids.append(int(user_id))
            item_ids.append(int(item_id))
            labels.append(float(prob))
        
        # 负样本：用户未交互过的物品（平衡采样）
        all_items = set(range(item_num))
        neg_items = all_items - set(group["item"].values)
        if len(neg_items) > 0:
            sample_size = min(len(group["item"].values), len(neg_items))
            neg_samples = np.random.choice(sorted(neg_items), size=sample_size, replace=False)
            for neg_id in neg_samples:
                # 优先用LLM概率，无匹配时默认负样本概率0.0
                prob = llm_probs.get((user_id, neg_id), 0.0)
                user_ids.append(int(user_id))
                item_ids.append(int(neg_id))
                labels.append(float(prob))
    
    return user_ids, item_ids, labels


# =====================================
# 8. 主流程
# =====================================
def main(args):
    # 设备初始化
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    set_seed(args.seed)

    # 路径设置（仅保留必要文件）
    root = os.path.join('data', args.dataset)
    train_interaction_path = os.path.join(root, args.train_interaction_file)  # 原始交互数据
    item_emb_path = os.path.join(root, args.item_emb_file)  # 物品嵌入
    user_emb_path = os.path.join(root, args.user_emb_file)  # 用户嵌入
    llm_json_path = os.path.join(root, args.llm_json_file)        # LLM软标签样本

    # 1. 加载LLM软标签（直接读取user_id和item_id）
    llm_probs = load_llm_probs(llm_json_path)

    # 2. 加载原始交互数据和嵌入文件
    train_interaction_df = pd.read_csv(train_interaction_path)
    item_origin_emb = torch.load(item_emb_path).float().to(device)  # (item_num, input_size)
    user_origin_emb = torch.load(user_emb_path).float().to(device)  # (user_num, input_size)
    if item_origin_emb.shape[1] != user_origin_emb.shape[1]:
        raise ValueError(
            f"Item/user embedding dimensions differ: "
            f"{item_origin_emb.shape[1]} vs {user_origin_emb.shape[1]}"
        )
    input_size = item_origin_emb.shape[1]
    if args.input_size is not None and args.input_size != input_size:
        print(f"[WARN] --input_size={args.input_size} does not match embedding dim={input_size}; using {input_size}.")

    # 3. 构建训练样本（正负样本+软标签）
    user_ids, item_ids, labels = build_interaction_samples(
        train_interaction_df,
        item_num=len(item_origin_emb),
        llm_probs=llm_probs,
    )

    # 4. 划分训练集/测试集（8:2）
    dataset = InteractionDataset(user_ids, item_ids, labels)
    train_size = int(0.8 * len(dataset))
    split_generator = torch.Generator().manual_seed(args.seed)
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, len(dataset) - train_size],
        generator=split_generator,
    )

    # 5. 创建数据加载器
    loader_generator = torch.Generator().manual_seed(args.seed)
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        generator=loader_generator,
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=pin_memory)

    # 6. 初始化模型、优化器和学习率调度器
    model = LlamaHead(input_size, args.hidden_size, args.output_size).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)  # 增加权重衰减防过拟合
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.epochs * len(train_loader)
    )

    # 7. 训练与验证
    best_acc, total_loss = 0.0, 0.0
    loss_fn = SoftBCEWithLogitsLoss()  # 软标签损失函数
    for epoch in range(1, args.epochs + 1):
        print(f"\n[INFO] Epoch {epoch}/{args.epochs}")
        # 训练轮次
        total_loss = train_loop(
            train_loader,
            model,
            loss_fn,
            optimizer,
            lr_scheduler,
            epoch,
            total_loss,
            device,
            user_origin_emb,
            item_origin_emb,
        )
        # 验证轮次
        valid_acc = test_loop(
            test_loader,
            model,
            device,
            user_origin_emb,
            item_origin_emb,
            mode="Valid",
        )

        # 保存最优模型（按验证集准确率）
        if valid_acc > best_acc:
            best_acc = valid_acc
            save_path = os.path.join(root, args.save_file)  # 保持默认文件名，兼容后续程序
            torch.save(model.state_dict(), save_path)
            print(f"[INFO] Saved best model (Acc: {best_acc:.4f}) to {save_path}")

    print(f"\n[INFO] Training finished! Best Valid Accuracy: {100 * best_acc:.2f}%")


# =====================================
# 9. 参数解析
# =====================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LLaMA SubTower with LLM soft labels")
    # 数据集参数
    parser.add_argument("--dataset", type=str, default="CiteULike", help="Dataset name (e.g., CiteULike, Movielens)")
    parser.add_argument("--train_interaction_file", type=str, default="warm_emb.csv")
    parser.add_argument("--item_emb_file", type=str, default="llm_item_content_emb.pt")
    parser.add_argument("--user_emb_file", type=str, default="llm_user_content_emb.pt")
    parser.add_argument("--llm_json_file", type=str, default="train_sample.json")
    parser.add_argument("--save_file", type=str, default="llama_head.bin")
    # 模型参数
    parser.add_argument("--input_size", type=int, default=None, help="LLM hidden size (inferred from embeddings by default)")
    parser.add_argument("--hidden_size", type=int, default=1024, help="MLP hidden layer size")
    parser.add_argument("--output_size", type=int, default=200, help="Output embedding dimension")
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=128, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate (AdamW)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # 设备参数
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (cuda:0 or cpu)")
    args = parser.parse_args()
    main(args)
