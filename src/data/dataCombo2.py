import csv

# Input files
first_file = input("Path to first CSV (has color, planttype, maintenance): ")
second_file = input("Path to second CSV (has meaning, occasion): ")
output_file = input("Path to output CSV: ")

# --- Read first CSV ---
first_rows = []
with open(first_file, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    first_fields = list(reader.fieldnames)
    for row in reader:
        first_rows.append({k: v.strip() for k, v in row.items()})

# --- Read second CSV ---
second_rows = []
with open(second_file, newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    second_fields = list(reader.fieldnames)
    for row in reader:
        second_rows.append({k: v.strip() for k, v in row.items()})

# --- Determine second-only fields ---
second_only_fields = [f for f in second_fields if f not in first_fields and f not in ("scientific_name", "name")]

# --- Combined field order for output ---
all_fields = first_fields + second_only_fields

# --- Build lookup from first CSV ---
sci_to_idx = {row["scientific_name"].lower(): i for i, row in enumerate(first_rows) if row.get("scientific_name")}
name_to_idx = {row["name"].lower(): i for i, row in enumerate(first_rows) if row.get("name")}

# --- Prepare output list ---
output_rows = []

# --- Merge ---
for s_row in second_rows:
    sci2 = s_row.get("scientific_name", "").lower()
    name2 = s_row.get("name", "").lower()

    # Find all first CSV rows that match this scientific name
    matched_indices = [i for i, r in enumerate(first_rows) if r.get("scientific_name","").lower() == sci2]

    if matched_indices:
        for idx in matched_indices:
            f_row = first_rows[idx]
            new_row = s_row.copy()
            # Fill missing fields from first CSV
            for field in ["planttype", "maintenance", "color"]:
                new_row[field] = f_row.get(field, "").strip()
            # Ensure all fields exist
            for f in all_fields:
                if f not in new_row:
                    new_row[f] = ""
            output_rows.append(new_row)
    else:
        # No match in first CSV, just append the second CSV row as-is
        new_row = s_row.copy()
        for f in all_fields:
            if f not in new_row:
                new_row[f] = ""
        output_rows.append(new_row)
# --- Write output CSV ---
with open(output_file, "w", newline="", encoding="utf-8") as out:
    writer = csv.DictWriter(out, fieldnames=all_fields)
    writer.writeheader()
    for row in output_rows:
        writer.writerow({f: row.get(f, "") for f in all_fields})

print(f"Done! Merged CSV saved to: {output_file}")