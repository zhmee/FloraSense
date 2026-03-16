import csv

#input files
scientific_file = input("Path to main dataset: ")       
output_file = input("Path to output")

# OPEN SCIENTIFIC COMMON NAMES FILE
with open(scientific_file, newline="", encoding="utf-8-sig") as input, \
     open(output_file, "w", newline="", encoding="utf-8") as output:

    reader = csv.DictReader(input)
    writer = csv.DictWriter(output, fieldnames=["common_name", "scientific_name", "color", "planttype", "maintenance", "meaning"])
    writer.writeheader()

    for row in reader:
        meaning = row.get("meaning")
        meaning = row.get("meaning", "").strip()
        if meaning:
            writer.writerow({
                "common_name": row["common_name"].strip(),
                "scientific_name": row["scientific_name"].strip(),
                "color": row["color"].strip(),
                "planttype": row["planttype"].strip(),
                "maintenance": row["maintenance"].strip(),
                "meaning": row["meaning"].strip()
            })

print(f"Done! Saved file to {output_file}")