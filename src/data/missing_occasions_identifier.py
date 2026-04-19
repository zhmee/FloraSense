import pandas as pd

CSV_PATH = "merged.csv"
OUTPUT_LIST_PATH = "missing_special_occasions.txt"  # optional

def is_missing(value):
    """Check if a field is empty, NaN, or just whitespace."""
    if pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False

def find_missing_special_occasions():
    df = pd.read_csv(CSV_PATH)

    # Filter rows where Special Occasions is missing
    missing_df = df[df["Special Occasions"].apply(is_missing)]

    # Extract names
    missing_names = (set)(missing_df["name"].tolist())

    # Print results
    print("Flowers missing 'Special Occasions':\n")
    for name in missing_names:
        print(f"- {name}")

    print(f"\nTotal: {len(missing_names)}")

    # Optional: save to file
    with open(OUTPUT_LIST_PATH, "w", encoding="utf-8") as f:
        for name in missing_names:
            f.write(name + "\n")

    print(f"\nSaved list to {OUTPUT_LIST_PATH}")

if __name__ == "__main__":
    find_missing_special_occasions()