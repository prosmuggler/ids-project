import glob, os
import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

for name in ["ciciot2023", "wsn_ds"]:
    print("\n" + "=" * 70)
    print(name.upper())
    print("=" * 70)

    files = sorted(glob.glob(f"data/raw/{name}/**/*.csv", recursive=True))
    total_gb = sum(os.path.getsize(f) for f in files) / 1e9
    print(f"{len(files)} CSV files, {total_gb:.2f} GB total")
    for f in files[:5]:
        print("  ", f, f"({os.path.getsize(f)/1e6:.1f} MB)")
    if not files:
        continue

    df = pd.read_csv(files[0], nrows=200_000)
    print(f"\nFirst file sample: {df.shape[0]} rows x {df.shape[1]} columns")
    print("\nColumns (repr shows hidden spaces):")
    print([c for c in df.columns])

    print("\nDtypes summary:")
    print(df.dtypes.value_counts())

    # guess the label column
    cands = [c for c in df.columns if c.strip().lower() in ("label", "attack type", "attack_type", "class")]
    print("\nLabel column candidates:", cands)
    for c in cands:
        print(f"\nValue counts for {c!r}:")
        print(df[c].value_counts().head(15))

    print("\nMissing values (top 5):")
    print(df.isna().sum().sort_values(ascending=False).head(5))
    print("\nHead:")
    print(df.head(3))
