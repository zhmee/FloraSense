"""
batch_meaning_summarizer.py
"""

import csv
import json
import logging
import os
import re
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_CSV  = "data/two.csv"
OUTPUT_CSV = "flowers_with_ir_summary_two.csv"
BATCH_SIZE = 20
SLEEP_BETWEEN_BATCHES = 1.5

COL_NAME        = "name"
COL_SCIENTIFIC  = "scientific_name"
COL_COLOR       = "color"
COL_PLANTTYPE   = "planttype"
COL_MAINTENANCE = "maintenance"
COL_MEANING     = "meaning"
COL_OCCASIONS   = "Special Occasions"

OUTPUT_FIELDS = [COL_NAME, COL_COLOR, "ir_summary", "occasions_summary"]


def _row_key(row: dict) -> str:
    return f"{row.get(COL_NAME, '').strip().lower()}|{row.get(COL_COLOR, '').strip().lower()}"


def _llm_client():
    api_key = os.getenv("SPARK_API_KEY")
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except Exception:
            pass
        api_key = os.getenv("SPARK_API_KEY")
    if not api_key:
        raise RuntimeError("SPARK_API_KEY not set.")
    from infosci_spark_client import LLMClient
    return LLMClient(api_key=api_key)


def _llm_text_response(client, messages):
    try:
        response = client.chat(messages, stream=False, show_thinking=False)
    except TypeError:
        response = client.chat(messages)
    return (response or {}).get("content", "").strip()


def _summarize_batch(client, flowers: list[dict]) -> dict[str, dict]:
    payload = [
        {
            "row_key": f["row_key"],
            "name": f["name"],
            "color": f["color"],
            "scientific_name": f["scientific_name"],
            "meaning": f["meaning"],
            "occasions": f["occasions"],
        }
        for f in flowers
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "Quickly summarize the meaning data for each flower so that it is more "
                "grammatically correct and easier to read. "
                "Do NOT add any new words or facts not already in the meaning text. "
                "Get rid of history lessons, cultural references, and repeated phrases — keep only the core flower meanings. "
                "Keep it to 3 sentences maximum, around 50 words total. "
                "For the occasions_summary, rewrite the occasions data into clean natural prose "
                "describing when and why this flower is given. "
                "Do not add any occasions not present in the original occasions text. "
                "Keep it to 2 sentence, around 30 words total. "
                "Never mention RAG, IR, vectors, retrieval, database, matched keywords, score, or context. "
                "Each entry has a row_key — return it exactly as given so results can be matched back. "
                "Return JSON only, as an array: "
                "[{\"row_key\": \"exact row_key\", \"ir_summary\": \"rewritten meaning\", "
                "\"occasions_summary\": \"rewritten occasions\"}]"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=True),
        },
    ]

    content = _llm_text_response(client, messages)
    content = content.strip()

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                logger.warning("Could not parse LLM response for batch.")
                return {}
        else:
            logger.warning("No JSON array found in LLM response.")
            return {}

    if not isinstance(parsed, list):
        return {}

    return {
        item["row_key"]: {
            "ir_summary": item.get("ir_summary", ""),
            "occasions_summary": item.get("occasions_summary", ""),
        }
        for item in parsed
        if isinstance(item, dict) and item.get("row_key")
    }


def main():
    client = _llm_client()

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.info(f"Loaded {len(rows)} rows from {INPUT_CSV}")

    for row in rows:
        row["_row_key"] = _row_key(row)

    # Load any existing progress
    results: dict[str, dict] = {}
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = f"{row.get(COL_NAME, '').strip().lower()}|{row.get(COL_COLOR, '').strip().lower()}"
                ir_summary = row.get("ir_summary", "").strip()
                occasions_summary = row.get("occasions_summary", "").strip()
                if ir_summary:
                    results[key] = {
                        "ir_summary": ir_summary,
                        "occasions_summary": occasions_summary,
                    }
        logger.info(f"Resuming — {len(results)} rows already done.")

    def _checkpoint():
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    COL_NAME: row.get(COL_NAME, ""),
                    COL_COLOR: row.get(COL_COLOR, ""),
                    "ir_summary": results.get(row["_row_key"], {}).get("ir_summary", ""),
                    "occasions_summary": results.get(row["_row_key"], {}).get("occasions_summary", ""),
                })

    batches = [rows[i:i + BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    for batch_index, batch in enumerate(batches):
        flowers = [
            {
                "row_key": row["_row_key"],
                "name": row.get(COL_NAME, ""),
                "color": row.get(COL_COLOR, ""),
                "scientific_name": row.get(COL_SCIENTIFIC, ""),
                "meaning": row.get(COL_MEANING, ""),
                "occasions": row.get(COL_OCCASIONS, ""),
            }
            for row in batch
            if row.get(COL_NAME) and row.get(COL_MEANING)
            and row["_row_key"] not in results
        ]

        if not flowers:
            logger.info(f"Batch {batch_index + 1}/{len(batches)} already done, skipping.")
            continue

        logger.info(f"Batch {batch_index + 1}/{len(batches)}...")
        batch_results = _summarize_batch(client, flowers)
        results.update(batch_results)
        logger.info(f"  Summarized: {[f['name'] + '/' + f['color'] for f in flowers]}")

        _checkpoint()
        logger.info(f"  Checkpoint saved.")

        if batch_index < len(batches) - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    logger.info(f"Done. Written to {OUTPUT_CSV}")
    missing = [
        f"{row.get(COL_NAME)}/{row.get(COL_COLOR)}"
        for row in rows
        if not results.get(row["_row_key"])
    ]
    if missing:
        logger.warning(f"Missing summaries for: {missing}")


if __name__ == "__main__":
    main()