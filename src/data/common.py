import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

MIN_CLASS_ROWS = 10  # classes smaller than this can't be split safely


def load_cfg():
    with open("configs/base.yaml") as f:
        return yaml.safe_load(f)


def clean_rows(df, log):
    """Row-level cleaning only (safe to do before splitting)."""
    log["rows_start"] = len(df)

    df = df.replace([np.inf, -np.inf], np.nan)
    n = len(df)
    df = df.dropna()
    log["dropped_nan_or_inf"] = n - len(df)

    n = len(df)
    df = df.drop_duplicates()  # done pre-split so identical rows can't leak across splits
    log["dropped_duplicates"] = n - len(df)

    log["rows_after_cleaning"] = len(df)
    return df.reset_index(drop=True)


def drop_tiny_classes(df, label_col, log):
    counts = df[label_col].value_counts()
    tiny = counts[counts < MIN_CLASS_ROWS].index.tolist()
    log["dropped_tiny_classes"] = tiny
    return df[~df[label_col].isin(tiny)].reset_index(drop=True)


def stratified_split(df, label_col, cfg):
    seed = cfg["seed"]
    tr, va, te = cfg["split"]["train"], cfg["split"]["val"], cfg["split"]["test"]
    train, rest = train_test_split(
        df, test_size=1 - tr, stratify=df[label_col], random_state=seed
    )
    val, test = train_test_split(
        rest, test_size=te / (va + te), stratify=rest[label_col], random_state=seed
    )
    return (x.reset_index(drop=True) for x in (train, val, test))


def drop_constant_cols(train, val, test, feature_cols, log):
    """Constant columns are detected on TRAIN only, then removed everywhere."""
    const = [c for c in feature_cols if train[c].nunique() <= 1]
    log["dropped_constant_columns"] = const
    return (
        train.drop(columns=const),
        val.drop(columns=const),
        test.drop(columns=const),
        [c for c in feature_cols if c not in const],
    )


def save_all(out_dir, train, val, test, feature_cols, log, label_cols):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, d in [("train", train), ("val", val), ("test", test)]:
        d.to_parquet(out / f"{name}.parquet", index=False)
        log[f"rows_{name}"] = len(d)
    log["n_features"] = len(feature_cols)
    log["feature_columns"] = feature_cols
    with open(out / "cleaning_log.json", "w") as f:
        json.dump(log, f, indent=2)

    counts = pd.concat(
        {n: d["attack_type"].value_counts() for n, d in
         [("train", train), ("val", val), ("test", test)]}, axis=1
    ).fillna(0).astype(int)
    counts.to_csv(out / "class_counts.csv")
    print(f"\nSaved to {out}/")
    print(counts.to_string())
