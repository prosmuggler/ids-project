import pandas as pd

from src.data.common import (load_cfg, clean_rows, drop_tiny_classes,
                             stratified_split, drop_constant_cols, save_all)

# identifiers: using them would let a model memorise nodes/rounds instead of behaviour
ID_COLS = ["id", "Time", "who CH"]


def main():
    cfg = load_cfg()
    df = pd.read_csv("data/raw/wsn_ds/WSN-DS.csv")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={"Expaned Energy": "Expended_Energy", "Attack type": "attack_type"})
    df["attack_type"] = df["attack_type"].str.strip()

    log = {"dataset": "WSN-DS", "dropped_identifier_columns": ID_COLS}
    df = df.drop(columns=ID_COLS)

    df["category"] = df["attack_type"]  # WSN-DS has only one level of attack naming
    df["binary"] = (df["attack_type"] != "Normal").astype("int8")

    label_cols = ["binary", "category", "attack_type"]
    feature_cols = [c for c in df.columns if c not in label_cols]

    df = clean_rows(df, log)
    df = drop_tiny_classes(df, "attack_type", log)

    train, val, test = stratified_split(df, "attack_type", cfg)
    train, val, test, feature_cols = drop_constant_cols(train, val, test, feature_cols, log)

    print("Cleaning log:")
    for k, v in log.items():
        if k != "feature_columns":
            print(f"  {k}: {v}")
    save_all("data/processed/wsn_ds", train, val, test, feature_cols, log, label_cols)


if __name__ == "__main__":
    main()
