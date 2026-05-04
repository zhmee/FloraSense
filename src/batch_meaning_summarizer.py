"""
batch_meaning_summarizer.py
"""

import csv
import json
import logging
import os
import re
import time
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INPUT_CSV  = "data/one.csv"
OUTPUT_CSV = "flowers_with_ir_summary.csv"
BATCH_SIZE = 10
SLEEP_BETWEEN_BATCHES = 1.5

COL_NAME        = "name"
COL_SCIENTIFIC  = "scientific_name"
COL_COLOR       = "color"
COL_PLANTTYPE   = "planttype"
COL_MAINTENANCE = "maintenance"
COL_MEANING     = "meaning"
COL_OCCASIONS   = "Special Occasions"

OUTPUT_FIELDS = [COL_NAME, "ir_summary", "occasions_summary"]


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]+", " ", (text or "").lower()).strip()


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


def _group_rows_by_name(rows: list[dict]) -> dict[str, list[dict]]:
    """Group all color variants of the same flower together."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        name = row.get(COL_NAME, "").strip()
        if name:
            groups[name].append(row)
    return dict(groups)


def _summarize_batch(client, flower_groups: list[dict]) -> dict[str, dict]:
    """
    Each item in flower_groups is:
    {
        "name": "Tulip",
        "variants": [
            {"color": "yellow", "meaning": "...", "occasions": "..."},
            {"color": "white",  "meaning": "...", "occasions": "..."},
        ]
    }
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You receive a list of flowers. Each flower has one or more color variants, "
                "each with its own meaning and occasions text. "
                "Your job is to write ONE combined ir_summary and ONE combined occasions_summary "
                "per flower (not per color variant). "
                "\n\n"
                "MEANING RULES:\n"
                "- If all color variants share the same core meaning, write one general summary. "
                "Do not mention colors at all in that case.\n"
                "- If different colors have meaningfully different meanings, briefly mention each "
                "color's meaning in one flowing sentence, e.g. "
                "'Red varieties symbolize passion, while white ones convey purity and pink express gratitude.'\n"
                "- Do NOT add any facts not present in the original meaning text.\n"
                "- Remove history lessons, cultural references, and repeated phrases.\n"
                "- Keep it to 3-4 sentences, around 50 words total.\n"
                "\n"
                "OCCASIONS RULES:\n"
                "- Combine all color variants' occasions into one natural summary.\n"
                "- Do not repeat the same occasion multiple times.\n"
                "- Do not add occasions not present in the original text.\n"
                "- Keep it to 1-2 sentences, around 30 words total.\n"
                "\n"
                "Return JSON only as an array, one entry per flower:\n"
                "[{\"name\": \"exact flower name\", \"ir_summary\": \"...\", "
                "\"occasions_summary\": \"...\"}]"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(flower_groups, ensure_ascii=True),
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
        item["name"]: {
            "ir_summary": item.get("ir_summary", ""),
            "occasions_summary": item.get("occasions_summary", ""),
        }
        for item in parsed
        if isinstance(item, dict) and item.get("name")
    }


def main():
    client = _llm_client()

    with open(INPUT_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    logger.info(f"Loaded {len(rows)} rows from {INPUT_CSV}")

    groups = _group_rows_by_name(rows)
    flower_names = list(groups.keys())
    logger.info(f"Found {len(flower_names)} unique flowers across {len(rows)} color variants")

    # load existing progress
    results: dict[str, dict] = {}
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get(COL_NAME, "").strip()
                ir_summary = row.get("ir_summary", "").strip()
                occasions_summary = row.get("occasions_summary", "").strip()
                if name and ir_summary:
                    results[name] = {
                        "ir_summary": ir_summary,
                        "occasions_summary": occasions_summary,
                    }
        logger.info(f"Resuming — {len(results)} flowers already done.")

    def _checkpoint():
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            for name, result in results.items():
                writer.writerow({
                    COL_NAME: name,
                    "ir_summary": result.get("ir_summary", ""),
                    "occasions_summary": result.get("occasions_summary", ""),
                })

    # build batches of flower groups (not individual rows)
    pending_names = [n for n in flower_names if n not in results]
    batches = [pending_names[i:i + BATCH_SIZE] for i in range(0, len(pending_names), BATCH_SIZE)]

    for batch_index, batch_names in enumerate(batches):
        flower_groups = [
            {
                "name": name,
                "variants": [
                    {
                        "color": row.get(COL_COLOR, "").strip(),
                        "meaning": row.get(COL_MEANING, "").strip(),
                        "occasions": row.get(COL_OCCASIONS, "").strip(),
                    }
                    for row in groups[name]
                    if row.get(COL_MEANING, "").strip()
                ],
            }
            for name in batch_names
        ]
        # skip flowers with no meaning data
        flower_groups = [fg for fg in flower_groups if fg["variants"]]

        if not flower_groups:
            logger.info(f"Batch {batch_index + 1}/{len(batches)} — nothing to do, skipping.")
            continue

        logger.info(f"Batch {batch_index + 1}/{len(batches)} — {[fg['name'] for fg in flower_groups]}")
        batch_results = _summarize_batch(client, flower_groups)
        results.update(batch_results)

        _checkpoint()
        logger.info(f"  Checkpoint saved. {len(results)}/{len(flower_names)} flowers done.")

        if batch_index < len(batches) - 1:
            time.sleep(SLEEP_BETWEEN_BATCHES)

    logger.info(f"Done. Written to {OUTPUT_CSV}")
    missing = [n for n in flower_names if n not in results]
    if missing:
        logger.warning(f"Missing summaries for: {missing}")


if __name__ == "__main__":
    main()