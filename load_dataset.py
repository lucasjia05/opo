import os, json, random
from datasets import load_dataset, get_dataset_config_names

IGNORE_CONFIGS = {"all", "auxiliary_train"}

def dump_split(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            if "answer" not in r:
                print(r)
            obj = {
                "label": r["answer"],   # 0 -> A, 1 -> B, ...
                "text": r["question"],
                "choices": r["choices"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main(out_dir="data/mmlu", seed=42):
    subjects = [s for s in get_dataset_config_names("cais/mmlu") if s not in IGNORE_CONFIGS]
    print(f"Downloading {len(subjects)} subjects → {out_dir}")

    random.seed(seed)

    for subj in subjects:
        print(" -", subj)
        ds = load_dataset("cais/mmlu", subj)
        subj_dir = os.path.join(out_dir, subj)
        os.makedirs(subj_dir, exist_ok=True)

        # collect ALL rows from ALL splits that exist
        all_rows = []
        for split in ["train", "dev", "validation", "test"]:
            if split in ds:
                all_rows.extend(ds[split])

        if not all_rows:
            print(f"   (no rows for {subj}, skipping)")
            continue

        # shuffle and split 50/50
        random.shuffle(all_rows)
        n = len(all_rows)
        mid = n // 2  # first half train, second half test

        train_rows = all_rows[:mid]
        test_rows = all_rows[mid:]

        dump_split(train_rows, os.path.join(subj_dir, "train.jsonl"))
        dump_split(test_rows, os.path.join(subj_dir, "test.jsonl"))

if __name__ == "__main__":
    main()
