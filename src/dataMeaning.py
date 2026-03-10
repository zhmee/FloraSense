import csv

main_file = input("Enter in the main file: ")
meanings_file = input("Enter file with meanings: ")
output_file = input("Enter file for output: ")
# List of meanings
meanings = []

with open(meanings_file, newline="", encoding="utf-8-sig") as meaning_file:
    reader = csv.DictReader(meaning_file)

    for row in reader:
        sci_name = row["scientific_name"].strip().lower()
        meaning = row["meaning"].strip().lower()

        color = row.get("color", "")
        color = color.strip().lower() if color else ""

        meanings.append({
            "scientific_name": sci_name,
            "color": color,
            "meaning": meaning
        })

# Write output
with open(main_file, newline="", encoding="utf-8-sig") as main, \
     open(output_file, "w", newline="", encoding="utf-8") as output:

    reader = csv.DictReader(main)
    writer = csv.DictWriter(
        output,
        fieldnames=["common_name", "scientific_name", "color", "planttype", "maintenance", "meaning"]
    )

    writer.writeheader()

    for row in reader:
        sci_name = row["scientific_name"].strip().lower()
        color = row["color"].strip().lower()

        found_meaning = ""

        # Going through the flower-colors-meanings list
        for m in meanings:
            if sci_name == m["scientific_name"]:
                if m["color"]:
                    color_match = color == m["color"]
                else:
                    color_match = True

                if color_match:
                    found_meaning = m["meaning"]
                    break

        writer.writerow({
            "common_name": row["common_name"].lower().strip(),
            "scientific_name": row["scientific_name"].lower().strip(),
            "color": row["color"].lower().strip(),
            "planttype": row.get("planttype", "").lower().strip(),
            "maintenance": row.get("maintenance", "").lower().strip(),
            "meaning": found_meaning
        })

print(f"Done! Saved output to {output_file}")