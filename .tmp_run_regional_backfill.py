from __future__ import annotations

import json
import time
from pathlib import Path

import requests


WINDOWS = (
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-02", "2024-12-31"),
    ("2025", "2025-01-01", "2025-12-31"),
    ("2026", "2026-01-01", "2026-09-22"),
)
INPUT_PATH = Path(".tmp_expand_regional_series_result.json")
OUTPUT_PATH = Path(".tmp_regional_backfill_result.json")


def save(result: dict) -> None:
    OUTPUT_PATH.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    prepared = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    ingredient_codes = prepared["ingredient_codes"]
    result = {
        "started_at_epoch": time.time(),
        "ingredient_count": len(ingredient_codes),
        "calls": [],
    }
    total_calls = len(ingredient_codes) * len(WINDOWS)
    call_number = 0

    for ingredient_code in ingredient_codes:
        for year, start_date, end_date in WINDOWS:
            call_number += 1
            call = {
                "ingredient_code": ingredient_code,
                "year": year,
                "start_date": start_date,
                "end_date": end_date,
            }
            try:
                response = requests.post(
                    "http://127.0.0.1:8080/api/prices/kamis/collect/"
                    + ingredient_code,
                    params={"startDate": start_date, "endDate": end_date},
                    timeout=240,
                )
                call["http_status"] = response.status_code
                if response.ok:
                    payload = response.json()
                    call.update(
                        {
                            "target_count": payload["targetCount"],
                            "targets_succeeded": payload["targetsSucceeded"],
                            "targets_failed": payload["targetsFailed"],
                            "fetched_rows": payload["fetchedRows"],
                            "prices_inserted": payload["pricesInserted"],
                            "duplicates_skipped": payload["duplicatesSkipped"],
                            "out_of_range_rows_skipped": payload[
                                "outOfRangeRowsSkipped"
                            ],
                            "invalid_rows_skipped": payload["invalidRowsSkipped"],
                            "failed_targets": [
                                target
                                for target in payload["targets"]
                                if not target["success"]
                            ],
                        }
                    )
                else:
                    call["error"] = response.text[:1000]
            except Exception as exception:
                call["http_status"] = None
                call["error"] = str(exception)

            result["calls"].append(call)
            save(result)
            print(
                f"[{call_number}/{total_calls}] {ingredient_code} {year} "
                f"status={call.get('http_status')} "
                f"success={call.get('targets_succeeded', 0)} "
                f"failed={call.get('targets_failed', 0)} "
                f"inserted={call.get('prices_inserted', 0)}",
                flush=True,
            )

    result["finished_at_epoch"] = time.time()
    save(result)


if __name__ == "__main__":
    main()
