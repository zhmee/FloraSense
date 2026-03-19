import os
import csv

folder_path = "../../data_scraping/flower_texts"

# collects all txt files
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
csv_files.sort()

# Write the filenames to a CSV
output_path = os.path.join(folder_path, "flower_file_titles.csv")
with open(output_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename"]) 
    for filename in csv_files:
        writer.writerow([filename])

print(f"Done! Saved {len(csv_files)} CSV file titles to {output_path}")