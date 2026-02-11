#!/usr/bin/env python3
"""
Test BM25 keyword search for a single patient.

Run from project root:
    python src/retrieval/test_retrieval_single_patient.py \\
        --corpus-dir /path/to/xml/subset \\
        --cache-dir /path/to/bm25_cache \\
        --person-id 136055918 \\
        --query "cancer chemotherapy radiation"

Requires: pip install -e '.[retrieval]'
"""

import argparse
import csv
import sys
from pathlib import Path

from retrieval.format_events import format_retrieved_events
from retrieval.local_retriever import LocalPatientRetriever


def main():
    parser = argparse.ArgumentParser(
        description="Test BM25 keyword search for a single patient."
    )
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="/data/fries/datasets/vista_bench_ryan/thoracic_cohort_lumia",
        help="Directory containing patient XML files ({person_id}.xml)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        # default="/data/fries/users/rdcunha/bm25_cache",
        default="/home/rdcunha/vista_project/bm25_test",
        help="Directory for BM25 index cache",
    )
    parser.add_argument(
        "--person-id",
        type=str,
        required=True,
        help="Patient ID (e.g. 136055918)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="cancer chemotherapy radiation",
        help="Search query string (default: 'cancer chemotherapy radiation')",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Maximum number of results to return (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/rdcunha/vista_project/bm25_test",
        help="Output path: CSV file or directory (default: bm25_test; creates bm25_test/bm25_results.csv if dir)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date filter (YYYY-MM-DD or YYYY-MM-DD HH:MM). Results on or after this date.",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date filter (YYYY-MM-DD or YYYY-MM-DD HH:MM). Results on or before this date.",
    )
    args = parser.parse_args()

    print(f"Initializing retriever: corpus={args.corpus_dir}, cache={args.cache_dir}")
    retriever = LocalPatientRetriever(
        corpus_dir=args.corpus_dir,
        cache_dir=args.cache_dir,
    )

    range_info = ""
    if args.start_date or args.end_date:
        range_info = f", date range: {args.start_date or '...'} to {args.end_date or '...'}"
    print(f"\nSearching person_id={args.person_id}, query='{args.query}', max_results={args.max_results}{range_info}")
    results = retriever.search(
        person_id=args.person_id,
        query=args.query,
        max_results=args.max_results,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    print(f"\nFound {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        score = r.get("score", "")
        timestamp = r.get("timestamp", "")
        event_type = r.get("event_type", "")
        content = (r.get("content", "") or "")[:400]
        if len(str(r.get("content", ""))) > 400:
            content += "..."
        print(f"  [{i}] score={score} | {timestamp} | {event_type}")
        print(f"      {content}")
        print()

    # Format as patient timeline string and save to CSV
    timeline_str = format_retrieved_events(results)
    out_path = Path(args.output)
    if out_path.suffix.lower() != ".csv":
        out_path = out_path / "bm25_results.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["person_id", "query", "max_results", "num_results", "start_date", "end_date", "patient_timeline"])
        writer.writerow([
            args.person_id,
            args.query,
            args.max_results,
            len(results),
            args.start_date or "",
            args.end_date or "",
            timeline_str,
        ])
    print(f"Saved formatted timeline to {out_path}")


if __name__ == "__main__":
    main()
