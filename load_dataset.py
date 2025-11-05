import os, json
from datasets import load_dataset, get_dataset_config_names

def dump_split(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            obj = {
                "label": r["answer"],   # 0 -> A, 1 -> B, ...
                "text": r["question"],
                "choices": r["choices"],
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main(out_dir="data/mmlu"):
    subjects = [s for s in get_dataset_config_names("cais/mmlu")]
    print(f"Downloading {len(subjects)} subjects → {out_dir}")
    for subj in subjects:
        print(" -", subj)
        ds = load_dataset("cais/mmlu", subj)
        subj_dir = os.path.join(out_dir, subj)
        os.makedirs(subj_dir, exist_ok=True)
        # combine dev + validation
        train_rows = []
        for split in ["dev", "validation"]:
            if split in ds:
                train_rows.extend(ds[split])
        dump_split(train_rows, os.path.join(subj_dir, "train.jsonl"))
        # test set
        if "test" in ds:
            dump_split(ds["test"], os.path.join(subj_dir, "test.jsonl"))

if __name__ == "__main__":
    main2()
