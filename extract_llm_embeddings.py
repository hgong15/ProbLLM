import os
import pickle
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm


# ======================
# Text encoding function
# ======================
def encode_text_batch(texts, model, tokenizer, device="cuda"):
    """
    Encode a batch of texts using LLaMA2 + Adapter, and return sentence embeddings.

    Args:
        texts (list[str]): Input text list
        model: Pretrained LLaMA2 model
        tokenizer: Corresponding tokenizer
        device (str): "cuda" or "cpu"

    Returns:
        torch.Tensor: Embeddings of shape (batch_size, hidden_size)
    """
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    ).to(device)

    with torch.inference_mode():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden_size)
    mask = inputs["attention_mask"].to(hidden_states.dtype).unsqueeze(-1)
    # LLaMA is decoder-only: the first token is usually a BOS token and is nearly
    # text-invariant. Pool real, non-padding tokens so the embedding reflects the
    # whole prompt/content text.
    pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return pooled


def resolve_torch_dtype(name):
    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


# ======================
# Dataset definitions
# ======================
class ContentDataset(Dataset):
    """Dataset for item content (titles)."""
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        return self.dataframe.iloc[idx].title


class UserDataset(Dataset):
    """Dataset for user preference texts."""
    def __init__(self, user_list):
        self.user_list = user_list

    def __len__(self):
        return len(self.user_list)

    def __getitem__(self, idx):
        return self.user_list[idx]


# ======================
# Main function
# ======================
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenizer.pad_token = tokenizer.eos_token  # Ensure pad token exists
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        device_map="auto",
        torch_dtype=resolve_torch_dtype(args.torch_dtype),
    )
    if args.adapter_model_path is not None:
        model = PeftModel.from_pretrained(model, args.adapter_model_path)
    model.eval()

    # 2. Load item content data
    data_root = os.path.join("./data", args.dataset)
    item_path = os.path.join(data_root, "raw-data.csv")

    if args.dataset == "CiteULike":
        item_content = pd.read_csv(item_path, encoding="latin1")
    else:
        item_content = pd.read_csv(item_path)

    content_dataset = ContentDataset(item_content)
    content_loader = DataLoader(
        content_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 3. Compute item embeddings
    item_embeddings = []
    print("Extracting item embeddings...")
    for batch in tqdm(content_loader, desc="Items", unit="batch"):
        emb = encode_text_batch(batch, model, tokenizer, device=device)
        item_embeddings.append(emb.cpu())  # Store on CPU to save GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
    item_emb = torch.cat(item_embeddings, dim=0)

    # 4. Load user-side text data. The default is the historical preference
    # list used by item-cold experiments; user-cold experiments pass a profile
    # based context file here so strict-cold users are not encoded as default info.
    user_pref_path = os.path.join(data_root, args.user_text_file)
    with open(user_pref_path, "rb") as f:
        train_user_preference_list = pickle.load(f)

    user_dataset = UserDataset(train_user_preference_list)
    user_loader = DataLoader(
        user_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 5. Compute user embeddings
    user_embeddings = []
    print("🔹 Extracting user embeddings...")
    for batch in tqdm(user_loader, desc="Users", unit="batch"):
        emb = encode_text_batch(batch, model, tokenizer, device=device)
        user_embeddings.append(emb.cpu())
        if device == "cuda":
            torch.cuda.empty_cache()
    user_emb = torch.cat(user_embeddings, dim=0)

    # 6. Save results
    item_save_path = os.path.join(data_root, args.item_output_file)
    user_save_path = os.path.join(data_root, args.user_output_file)
    torch.save(item_emb, item_save_path)
    torch.save(user_emb, user_save_path)

    print(f"Item embeddings saved to {item_save_path}")
    print(f"User embeddings saved to {user_save_path}")


# ======================
# Command-line interface
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LLM embeddings for items and users.")
    parser.add_argument("--model_dir", type=str, default="../models/Llama-3.2-1B",
                        help="Path to base LLaMA2 model directory")
    parser.add_argument("--adapter_model_path", type=str, default="./weight/llama3-1b_seed42",
                        help="Path to adapter model (optional)")
    parser.add_argument("--dataset", type=str, default="CiteULike", help="Dataset name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for DataLoader")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--user_text_file", type=str, default="train_user_preference_list.pkl",
                        help="Pickle file containing one user text per user.")
    parser.add_argument("--user_output_file", type=str, default="llm_user_content_emb.pt",
                        help="Output filename for user embeddings under data/<dataset>.")
    parser.add_argument("--item_output_file", type=str, default="llm_item_content_emb.pt",
                        help="Output filename for item embeddings under data/<dataset>.")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="Torch dtype used when loading the base model")

    args = parser.parse_args()
    main(args)
