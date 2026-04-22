import csv
import re
import numpy as np

def build_vocab_from_csv(file_path):
    vocab = set()

    with open(file_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)

        for row in reader:
            for cell in row:
                # tokenize words (letters + numbers only)
                words = re.findall(r"\b\w+\b", cell.lower())
                vocab.update(words)

    print(f"Vocab size: {len(vocab)}")
    return vocab



def filter_glove(glove_path, vocab, output_path, max_extra=20000):
    kept = 0
    extra_kept = 0

    with open(glove_path, "r", encoding="utf-8") as g, \
         open(output_path, "w", encoding="utf-8") as out:
        
        for line in g:
            parts = line.split()
            word = parts[0]

            if word in vocab:
                out.write(line)
                kept += 1
            elif extra_kept < max_extra:
                # keep top common words as fallback
                out.write(line)
                extra_kept += 1

    print(f"Kept vocab words: {kept}")
    print(f"Kept extra common words: {extra_kept}")


def glove_to_numpy(glove_file, vec_out, vocab_out):
    words = []
    vectors = []

    with open(glove_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            words.append(parts[0])
            vectors.append([float(x) for x in parts[1:]])

    vectors = np.array(vectors, dtype=np.float32)

    np.save(vec_out, vectors)
    np.save(vocab_out, words)

    print(f"Saved vectors: {vec_out}")
    print(f"Saved vocab: {vocab_out}")
    print(f"Shape: {vectors.shape}")


# ----------------------------
# Main pipeline
# ----------------------------
if __name__ == "__main__":
    DATASET_PATH = "merged.csv"
    GLOVE_PATH = "glove.6B.100d.txt"
    FILTERED_GLOVE = "filtered_glove.txt"


    vocab = build_vocab_from_csv(DATASET_PATH)


    extra_words = {
        "love", "grief", "mourning", "joy", "friendship",
        "romance", "elegant", "delicate", "wild", "gentle",
        "happy", "mother", "father", "sister", "brother", 
        "grandfather", "grandmother",
    }
    vocab.update(extra_words)
    print(f"Vocab size after extras: {len(vocab)}")

    # Step 2
    filter_glove(GLOVE_PATH, vocab, FILTERED_GLOVE)

    # Step 3
    glove_to_numpy(FILTERED_GLOVE, "vectors.npy", "vocab.npy")