import csv

# Input files
names_file = input("Enter path to CSV with common name + scientific name: ")
colors_file = input("Enter path to CSV with common name + color: ")
output_file = input("Enter path for output CSV: ")

# scientific_name map to -> common_name
sci_to_common = {}
with open(names_file, newline="", encoding="utf-8-sig") as names_file:
    reader = csv.DictReader(names_file)
    for row in reader:
        scientific_name = row["scientific_name"].strip().lower()
        common_name = row["name"].strip()
        sci_to_common[scientific_name] = common_name

with open(colors_file, newline="", encoding="utf-8-sig") as colors_file, \
     open(output_file, "w", newline="", encoding="utf-8") as output:

    reader = csv.DictReader(colors_file)
    writer = csv.DictWriter(output, fieldnames=["common_name","scientific_name","color"])
    writer.writeheader()

    for row in reader:
        sci_name = row["scientific_name"].strip().lower()
        common_name = sci_to_common.get(sci_name.lower(), "")
        writer.writerow({
            "common_name": common_name,
            "scientific_name": sci_name,
            "color": row["color"].strip().lower()
        })

print(f"Done! File saved to {output_file}")