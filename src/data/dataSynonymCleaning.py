import csv

# Input files
synonyms_file = input("Enter path to CSV with synonyms: ")
synonyms_file = input("Enter path to CSV with synonyms: ")
output_file = input("Enter path for output CSV: ")

# Load common_name -> scientific_name mapping
common_to_sci = {}
with open(names_file, newline="", encoding="utf-8-sig") as nf:
    reader = csv.DictReader(nf)
    for row in reader:
        common_to_sci[row["name"].strip().lower()] = row["scientific_name"].strip()

# Merge
with open(meanings_file, newline="", encoding="utf-8-sig") as mf, \
     open(output_file, "w", newline="", encoding="utf-8") as of:

    reader = csv.DictReader(mf)
    fieldnames = ["name", "scientific_name", "color", "meaning"]
    writer = csv.DictWriter(of, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        common_name = row["name"].strip()
        color = row.get("color", "").strip()  # may be empty, put ""
        sci_name = common_to_sci.get(common_name.lower(), "") # may be empty, put ""

        writer.writerow({
            "name": common_name,
            "scientific_name": sci_name,
            "color": color,
            "meaning": row["meaning"].strip()
        })

print(f"Done! Output saved to {output_file}")