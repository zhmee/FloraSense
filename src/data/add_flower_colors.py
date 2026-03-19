import csv
import re

#THIS SCRIPT TAKES IN FLOWER NAMES AND COLOR DESCRIPTIONS 
#THEN MAKES A FLOWER COLOR PAIR FOR EVERY SINGLE COLOR VARIATION

# File paths
flower_file = input("Path to Flower-Color CSV: ")
color_file = input("Path to used colors CSV: ")
output_file = input("Path for output: ")

# Load all provided colors into a set 
#USES COLOR FILE
with open(color_file, newline="", encoding="utf-8-sig") as colors:
    color_reader = csv.reader(colors)
    next(color_reader)
    color_set = set(row[0].lower() for row in color_reader if row[0].strip())

#pattern: remove parentheses in Name
paren_pattern = re.compile(r"\s*\([^)]*\)")

#pattern: split color description into words
word_pattern = re.compile(r'\b\w+\b')

#opens the flower-color file
with open(flower_file, newline="", encoding="utf-8-sig") as input_flower, \
     open(output_file, "w", newline="", encoding="utf-8") as output_file:

    
    reader = csv.DictReader(input_flower)
    # Fields included
    writer = csv.DictWriter(output_file, fieldnames=["name", "color", "planttype", "maintenance"])
    writer.writeheader()

    for row in reader:
        raw_name = row["name"]
        name_no_parens = paren_pattern.sub("", raw_name)
        name_cleaned = name_no_parens.strip().lower()
        color_description = row["color"].lower()

        planttype = row.get("planttype", "").strip()
        maintenance = row.get("maintenance", "").strip()

        # Extract words from description
        colors = word_pattern.findall(color_description)
        matched_colors = set()

        for color in colors:
            if color in color_set:
                matched_colors.add(color)

        #one row per color
        for color in matched_colors:
            writer.writerow({
                "name": name_clean,
                "color": color,
                "planttype": planttype,
                "maintenance": maintenance
            })

print(f"Done! Saved to {output_file}")