import csv

# Input files
names_file = input("Enter path to CSV with common name + scientific name: ")
data_file = input("Enter path to CSV with common name (and other columns): ")
output_file = input("Enter path for output CSV: ")

# Load common_name -> scientific_name mapping
common_to_sci = {}
with open(names_file, newline="", encoding="utf-8-sig") as nf:
    reader = csv.DictReader(nf)
    for row in reader:
        common_to_sci[row["name"].strip().lower()] = row.get("scientific_name", "").strip()

# Merge
with open(data_file, newline="", encoding="utf-8-sig") as df, \
     open(output_file, "w", newline="", encoding="utf-8") as of:

    reader = csv.DictReader(df)
    # Add scientific_name column if it doesn't exist
    fieldnames = reader.fieldnames + ["scientific_name"] if "scientific_name" not in reader.fieldnames else reader.fieldnames
    writer = csv.DictWriter(of, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        common_name = row["name"].strip()
        row["scientific_name"] = common_to_sci.get(common_name.lower(), "")
        writer.writerow(row)

print(f"Done! Output saved to {output_file}")