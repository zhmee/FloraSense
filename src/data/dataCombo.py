import csv

#input files
scientific_file = input("Path to Scientific_name, color, planttype, maintence: ")
common_file = input("Path to Scientific-Common Names: ")        
output_file = input("Path to output")


# Load scientific info into a dictionary keyed by (scientific_name, color)
sci_info = {}
with open(scientific_file, newline="", encoding="utf-8-sig") as sf:
    reader = csv.DictReader(sf)
    for row in reader:
        key = (row["scientific_name"].strip().lower(), row["color"].strip().lower())
        planttype = row.get("planttype", "").strip()
        maintenance =  row.get("maintenance", "").strip()
        sci_info[key] = {
            "planttype": planttype,
            "maintenance": maintenance
        }

# OPEN SCIENTIFIC COMMON NAMES FILE
with open(common_file, newline="", encoding="utf-8-sig") as input, \
     open(output_file, "w", newline="", encoding="utf-8") as output:

    reader = csv.DictReader(input)
    writer = csv.DictWriter(output, fieldnames=["common_name", "scientific_name", "color", "planttype", "maintenance"])
    writer.writeheader()

    for row in reader:
        sci_name = row["scientific_name"].strip()
        color = row["color"].strip()
        key = (sci_name.lower(), color.lower()) #im matching based on the name AND color since some plants have name and color meanings together

        info = sci_info.get(key)  # None if not found (aka can't find the color and scientific name)
        if info:
            planttype = info.get("planttype", "")
            maintenance = info.get("maintenance", "")
        else:
            planttype = ""
            maintenance = ""

        writer.writerow({
            "common_name": row["common_name"].strip(),
            "scientific_name": sci_name,
            "color": color,
            "planttype": planttype,
            "maintenance": maintenance
        })

print(f"Done! Saved file to {output_file}")