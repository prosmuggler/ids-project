import glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.common import (load_cfg, clean_rows, drop_tiny_classes,
                             stratified_split, drop_constant_cols, save_all)

KEEP_ALL_MB = 30  # attack folders smaller than this keep 100% of rows (rare classes)


def attack_name(path: Path) -> str:
    parts = path.parts
    if "CSV" in parts:
        i = parts.index("CSV")
        if len(parts) > i + 2:
            return parts[i + 1]
    return path.parent.name


def to_category(name: str):
    n = name.lower()
    if "benign" in n: return "Benign"
    if "ddos" in n: return "DDoS"
    if n.startswith("dos"): return "DoS"
    if "mirai" in n: return "Mirai"
    if "recon" in n or "vulnerab" in n: return "Recon"
    if "spoof" in n or "mitm" in n: return "Spoofing"
    if "brute" in n: return "BruteForce"
    if any(k in n for k in ["sql", "xss", "command", "upload", "browser", "backdoor"]):
        return "Web"
    return None


def main():
    cfg = load_cfg()
    seed, frac = cfg["seed"], cfg["sample_fraction"]
    files = sorted(Path(p) for p in glob.glob("data/raw/ciciot2023/**/*.csv", recursive=True))

    groups = defaultdict(list)
    for f in files:
        groups[attack_name(f)].append(f)

    # check the label mapping BEFORE doing any heavy work
    unmapped = [a for a in groups if to_category(a) is None]
    print(f"{len(groups)} attack folders found:")
    for a in sorted(groups):
        print(f"  {a:35s} -> {to_category(a)}")
    if unmapped:
        raise SystemExit(f"\nUnmapped folders (add them to to_category): {unmapped}")

    frames = []
    for attack, fl in tqdm(groups.items(), desc="folders"):
        size_mb = sum(f.stat().st_size for f in fl) / 1e6
        f_use = 1.0 if size_mb < KEEP_ALL_MB else frac
        cat = to_category(attack)
        atk = "Benign" if cat == "Benign" else attack
        for f in fl:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            if f_use < 1.0:
                df = df.sample(frac=f_use, random_state=seed)
            num = df.select_dtypes(include=[np.number]).columns
            df[num] = df[num].astype("float32")
            df["category"] = cat
            df["attack_type"] = atk
            frames.append(df)

    df = pd.concat(frames, ignore_index=True)
    del frames
    df["binary"] = (df["category"] != "Benign").astype("int8")
    print(f"\nSampled rows: {len(df):,}")

    label_cols = ["binary", "category", "attack_type"]
    feature_cols = [c for c in df.columns if c not in label_cols]

    log = {"dataset": "CICIoT2023", "sample_fraction": frac,
           "keep_all_mb_threshold": KEEP_ALL_MB}
    df = clean_rows(df, log)
    df = drop_tiny_classes(df, "attack_type", log)

    train, val, test = stratified_split(df, "attack_type", cfg)
    train, val, test, feature_cols = drop_constant_cols(train, val, test, feature_cols, log)

    print("\nCleaning log:")
    for k, v in log.items():
        if k != "feature_columns":
            print(f"  {k}: {v}")
    save_all("data/processed/ciciot2023", train, val, test, feature_cols, log, label_cols)


if __name__ == "__main__":
    main()
