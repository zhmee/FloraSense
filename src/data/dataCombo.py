import csv

# Input files
first_file = input("Path to first CSV (has color, meaning, etc.): ")
second_file = input("Path to second CSV (has meaning, occasion): ")
output_file = input("Path to output CSV: ")

#FIRST CSV
first_rows = []
with open(first_file, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    first_fields = list(reader.fieldnames)
    for row in reader:
        first_rows.append({k: v.strip() for k, v in row.items()})

#SECOND CSV
second_rows = []
with open(second_file, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    second_fields = list(reader.fieldnames)
    for row in reader:
        second_rows.append({k: v.strip() for k, v in row.items()})

# Determine new columns from second CSV not in first CSV
second_only_fields = [
    f for f in second_fields
    if f not in first_fields and f not in ("scientific_name", "common_name")
]

# OR all columns from SECOND and FIRST CSV
all_fields = first_fields + second_only_fields

#Lookup table using first csv (scientific_name and common_name)
sci_to_idx = {}
cname_to_idx = {}

for i, row in enumerate(first_rows):
    sci = row.get("scientific_name", "").lower()
    cname = row.get("common_name", "").lower()
    if sci:
        sci_to_idx.setdefault(sci, []).append(i)
    if cname:
        cname_to_idx.setdefault(cname, []).append(i)

# Track which rows got matched
matched_first_indices = set()

# Output rows start as copies of first_rows (with blank new columns)
output_rows = []
for row in first_rows:
    new_row = dict(row)
    for f in second_only_fields:
        new_row[f] = ""
    output_rows.append(new_row)

# These are the second rows that were unmatched (need to append at end)
unmatched_second_rows = []

for s_row in second_rows:
    sci2 = s_row.get("scientific_name", "").lower()
    cname2 = s_row.get("common_name", "").lower()

    # match on scientific names first, the common names
    matched_indices = []
    if sci2 and sci2 in sci_to_idx:
        matched_indices = sci_to_idx[sci2]
    elif cname2 and cname2 in cname_to_idx:
        matched_indices = cname_to_idx[cname2]

    if matched_indices:
        for idx in matched_indices:
            matched_first_indices.add(idx)
            # Merge the meanings from the first and the second
            existing_meaning = output_rows[idx].get("meaning", "").strip()
            new_meaning = s_row.get("meaning", "").strip()
            if existing_meaning and new_meaning:
                output_rows[idx]["meaning"] = f"{existing_meaning}; {new_meaning}"
            elif new_meaning:
                output_rows[idx]["meaning"] = new_meaning

            # Add the second only csv fields (occassions)
            for f in second_only_fields:
                val = s_row.get(f, "").strip()
                existing = output_rows[idx].get(f, "").strip()
                if existing and val:
                    output_rows[idx][f] = f"{existing}; {val}"
                elif val:
                    output_rows[idx][f] = val
    else:
        unmatched_second_rows.append(s_row)

# Append all unmatched second csv rows
for s_row in unmatched_second_rows:
    new_row = {}
    for f in first_fields:
        if f == "scientific_name":
            new_row[f] = s_row.get("scientific_name", "")
        elif f == "common_name":
            new_row[f] = s_row.get("common_name", "")
        elif f == "meaning":
            new_row[f] = s_row.get("meaning", "")
        else:
            #second csv does not have information for that particular piece of information, so add " "
            new_row[f] = " "  
    for f in second_only_fields:
        new_row[f] = s_row.get(f, "")
    output_rows.append(new_row)

# writing the output
with open(output_file, "w", newline="", encoding="utf-8") as out:
    writer = csv.DictWriter(out, fieldnames=all_fields)
    writer.writeheader()
    for row in output_rows:
        writer.writerow({f: row.get(f, "") for f in all_fields})

print(f"Done! Merged CSV saved to: {output_file}")