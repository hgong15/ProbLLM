# Data Preparation Notes

This directory documents how to prepare local dataset folders for ProbLLM. To run the pipeline, place prepared datasets under `data/` using the layout described below.

## Dataset Sources

The paper uses normalized interaction files derived from public recommendation benchmarks. Dataset licenses and redistribution terms remain those of the upstream datasets.

| Dataset folder used by scripts | Upstream source | Processed data characteristics | Side information used by ProbLLM |
| --- | --- | --- | --- |
| `CiteULike` | CiteULike benchmark data, commonly distributed through the [`citeulike-a`](https://github.com/js05212/citeulike-a) and [`citeulike-t`](https://github.com/js05212/citeulike-t) research mirrors. | 5,551 users, 16,980 articles, 204,986 user-article interactions. Mainly used for item cold-start recommendation. | Article titles and abstracts. The prepared `raw-data.csv` contains fields such as `doc.id`, `title`, `citeulike.id`, `raw.title`, and `raw.abstract`. |
| `ml-1m` | [GroupLens MovieLens-1M](https://grouplens.org/datasets/movielens/1m/). | 6,040 users, 3,883 movies, 1,000,210 explicit ratings in the paper's processed version. Used for both item and user cold-start recommendation. | Movie titles and genres. The prepared `raw-data.csv` contains `item`, `raw_item`, `title`, and `genres`. |
| `book-crossing` | [Book-Crossing (BX)](https://grouplens.org/datasets/book-crossing/), collected by Cai-Nicolas Ziegler and distributed by GroupLens/public mirrors. | 278,858 users, 271,379 books, 1,149,780 ratings. Used especially for user cold-start recommendation. | Book metadata such as title, author, and publisher, plus user attributes such as location and age when reproducing user-side prompt/content features. |
| `amazon23_appliances_item_y2023_rel50` | [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) from McAuley Lab, Appliances category; the dataset is also available through [Hugging Face](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023). | Post-cutoff audit dataset built from January 1 to December 31, 2023 interactions, keeping positive reviews with rating `>=4` before applying the temporal protocol and count filters. | Item title, category, feature, description, and detail text from Amazon Reviews 2023 metadata. The prepared `raw-data.csv` contains fields such as `title`, `categories`, `features`, `description`, `details`, and `metadata_text`. |

## Expected Local Layout

Each dataset directory should contain a normalized interaction file named after the dataset folder:

```text
data/
  CiteULike/
    CiteULike.csv
    raw-data.csv
  ml-1m/
    ml-1m.csv
    raw-data.csv
  book-crossing/
    book-crossing.csv
    raw-data.csv              # needed for text/user-side feature reproduction
  amazon23_appliances_item_y2023_rel50/
    amazon23_appliances_item_y2023_rel50.csv
    raw-data.csv
    id_mapping.json           # useful for tracing normalized ids to raw ids
```

The required interaction CSV has exactly the columns used by `data/split.py`:

```csv
user,item
0,15
0,42
1,7
```

Important conventions:

- `user` and `item` must be integer ids.
- Ids should be nonnegative and preferably contiguous after remapping, because the split code sets `user_num=max(user)+1` and `item_num=max(item)+1`.
- Duplicate `(user, item)` rows are dropped by `data/split.py`.
- For LLM prompts, candidate construction, and content embedding scripts, keep dataset-specific metadata in `raw-data.csv` or in the precomputed feature files expected by the corresponding script.

## Processing Protocol

The source code assumes a two-step split and conversion process.

1. Convert the public raw benchmark into the normalized `<dataset>.csv` file and optional metadata files. Keep an id mapping if you need to trace normalized ids back to raw user/item ids.
2. Run `data/split.py`. This creates the paper-style warm, warm-up, strict cold-start, validation, and test files.
3. Run `data/convert.py`. This builds `convert_dict.pkl`, neighbor arrays, entity group arrays, and `overall_val.csv`/`overall_test.csv` for downstream training and evaluation.
4. Generate SFT examples and LLM/content embeddings with the top-level scripts.
5. Build candidate pools, score candidate pairs with the LLM refiner, select pseudo-interactions, and rerun the downstream recommender.

Default item-side split example:

```bash
python data/split.py --datadir data --dataset CiteULike --seed 42 --cold_object item --warmup_k 5
python data/convert.py --datadir data --dataset CiteULike --seed 42 --cold_object item --protocol warmup --warmup_k 5
```

Default user-side split example:

```bash
python data/split.py --datadir data --dataset book-crossing --seed 42 --cold_object user --warmup_k 5
python data/convert.py --datadir data --dataset book-crossing --seed 42 --cold_object user --protocol warmup --warmup_k 5
```

## Split Semantics

For a chosen cold side (`--cold_object item` or `--cold_object user`), `data/split.py` partitions entities as follows:

- 80% warm entities,
- 10% strict cold-start entities,
- 10% warm-up entities,
- warm entities are split into train/validation/test interactions with the default `8:1:1` ratio,
- strict cold-start entities have no training support interactions and are evaluated through validation/test interactions,
- warm-up entities receive a small support set and are then evaluated on the remaining interactions,
- the default `k0` boundary is `--warmup_k 5`; warm-up support is sampled from `1..k0-1`, and entities with too few interactions are kept out of the warm-up group.

The script writes files such as:

```text
warm_train.csv
warm_val.csv
warm_test.csv
warmup_support.csv
warmup_val.csv
warmup_test.csv
cold_item.csv / cold_user.csv
cold_item_val.csv / cold_user_val.csv
cold_item_test.csv / cold_user_test.csv
overall_val.csv
overall_test.csv
split_meta.json
n_user_item.pkl
```

`data/convert.py` then stores the downstream protocol dictionary in `convert_dict.pkl`. The recommender and candidate-generation scripts read these generated files rather than the raw public dumps directly.
