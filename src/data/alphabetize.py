import csv

# Input/output files
input_file = input("Path to input CSV: ")
output_file = input("Path to output CSV: ")

# Column to sort by (default: first column)
sort_column = input("Column to sort by (leave blank for first column): ")

# --- Read CSV ---
with open(input_file, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = reader.fieldnames

# Determine sort column
if not sort_column:
    sort_column = fieldnames[0]

# --- Sort rows ---
rows_sorted = sorted(rows, key=lambda r: r.get(sort_column, "").lower())

# --- Write sorted CSV ---
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows_sorted)

print(f"Done! CSV sorted by '{sort_column}' saved to: {output_file}")