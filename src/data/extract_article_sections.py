import os
import re
import csv
import pandas as pd

# --- Paths ---
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, "..", "..", "data_scraping", "flower_texts")
folder_path = os.path.abspath(folder_path)

#CSV containing all flower files
flower_list_csv = os.path.join(folder_path, "flower_file_titles.csv")  
flower_files_df = pd.read_csv(flower_list_csv)
txt_files = flower_files_df['filename'].tolist()

# --- PATTERNS FROM CATEGORIES IN THE ARTICLE ---
meaning_patterns = [
    re.compile(r"What does the .*? flower mean\??", re.IGNORECASE),
    re.compile(r"What Does the .*? Flower Mean", re.IGNORECASE),
    re.compile(r"Meaning of the .*? Flower", re.IGNORECASE),
    re.compile(r".*? Flower meaning", re.IGNORECASE)
]

symbolism_patterns = [
    re.compile(r"Symbolism of the .*? Flower", re.IGNORECASE),
    re.compile(r"Symbolism of the .*? Flowers", re.IGNORECASE),
    re.compile(r"\.*? Flower Symbolism", re.IGNORECASE)
]

occasions_patterns = [
    re.compile(r"Special Occasions for .*? Flower", re.IGNORECASE),
    re.compile(r"Special Occasions for .*? Flowers", re.IGNORECASE),
    re.compile(r"Occasions for .*? Flowers", re.IGNORECASE)
]

# --- Filter out these words ---
filtered_words = {"a", "is", "there", "it", "would", "when", "the", "has", "had", "this", "to", "of", "what", "could", "can", "are", "that"}

# --- Preprocessing function ---
def preprocess_text(text):
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in filtered_words]
    return " ".join(words)

# --- Section extraction ---
def extract_section(text, patterns):
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        for pat in patterns:
            if pat.search(line):
                start_idx = i + 1
                break
        if start_idx is not None:
            break
    if start_idx is None:
        return ""

    section_lines = []
    for line in lines[start_idx:]:
        # Only take information BETWEEN headings
        if re.match(r"^[A-Z][A-Za-z\s]*$", line.strip()) or re.match(r"^[A-Z].*Flower", line.strip()):
            break
        section_lines.append(line)
    
    section_text = " ".join([l.strip() for l in section_lines if l.strip()])
    return preprocess_text(section_text)

#Get all flowers, and the meanings + occassions
all_flowers = {}

for filename in txt_files:
    file_path = os.path.join(folder_path, filename)
    # Flower name = everything BEFORE -flower-meaning.txt
    flower_name = re.split(r"-flower-meaning\.txt$", filename, flags=re.IGNORECASE)[0].replace("-", " ").title()

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

        meaning_text = extract_section(text, meaning_patterns)
        symbolism_text = extract_section(text, symbolism_patterns)
        occasions_text = extract_section(text, occasions_patterns)

        combined_text = " ".join(filter(None, [meaning_text, symbolism_text]))

        if combined_text or occasions_text:
            all_flowers[flower_name] = {
                "Meaning & Symbolism": combined_text,
                "Special Occasions": occasions_text
            }

# Save to output CSV
output_path = os.path.join(folder_path, "flowers_meaning_occasions.csv")
with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Flower", "Meaning & Symbolism", "Special Occasions"])
    for name, sections in sorted(all_flowers.items()):
        writer.writerow([name, sections["Meaning & Symbolism"], sections["Special Occasions"]])

print(f"Done! Saved {len(all_flowers)} flowers with sections to {output_path}")