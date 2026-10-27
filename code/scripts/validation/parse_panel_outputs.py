#!/usr/bin/env python3
"""
Parse the 3 judge text outputs per track into the two CSVs:
  outputs/pilot_eval/nli_panel_scores.csv      (108 rows)
  outputs/pilot_eval/bertscore_panel_categories.csv  (45 rows)

Usage:
  python scripts/validation/parse_panel_outputs.py \
      --raw-dir   panel_raw \
      --out-nli   outputs/pilot_eval/nli_panel_scores.csv \
      --out-bert  outputs/pilot_eval/bertscore_panel_categories.csv

Input files expected in --raw-dir:
  nli_claude.txt, nli_gemini.txt, nli_deepseek.txt
  bertscore_claude.txt, bertscore_gemini.txt, bertscore_deepseek.txt

Fails loudly if any judge file does not produce exactly the expected
number of pair_ids (36 for NLI, 15 for BERTScore).
"""
import argparse
import csv
import re
import sys
from pathlib import Path

JUDGES = ['claude', 'gemini', 'deepseek']

# Regex patterns: capture pair_id, then any text, then the final score/category.
# DOTALL so "." matches newlines (judges sometimes write multi-line bullets).
# Non-greedy so we stop at the first SCORE/CATEGORY after each pair_id.
NLI_PATTERN  = re.compile(r'\b(P\d{2})\b.*?SCORE:\s*(\d)', re.DOTALL)
BERT_PATTERN = re.compile(r'\b(BVAL-\d{3})\b.*?CATEGORY:\s*([A-D])', re.DOTALL)

# Pre-specified scenario → total pair counts
SCENARIO_COUNTS = {
    "pilot": {"nli": 36, "bert": 15},   # 36 NLI (P01..P36) + 15 BERTScore (BVAL-001..015)
    "main":  {"nli": 60, "bert": 25},   # 60 NLI (P01..P60) + 25 BERTScore (BVAL-001..025)
}


def parse_one(text, pattern, expected_pairs, judge_id, track):
    """Parse one judge's output; verify completeness."""
    matches = pattern.findall(text)
    # Keep ONLY the first occurrence of each pair_id (in case judge restated)
    seen = {}
    for pid, val in matches:
        if pid not in seen:
            seen[pid] = val
    missing = [p for p in expected_pairs if p not in seen]
    extra   = [p for p in seen if p not in expected_pairs]
    if missing:
        sys.exit(f'[{track} / {judge_id}] MISSING {len(missing)} pair_ids: {missing}')
    if extra:
        sys.exit(f'[{track} / {judge_id}] UNEXPECTED pair_ids: {extra}')
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scenario', choices=list(SCENARIO_COUNTS), default='pilot',
                    help='pilot = 36 NLI + 15 BERTScore; main = 60 NLI + 25 BERTScore')
    ap.add_argument('--raw-dir',  required=True, help='Directory with 6 .txt files')
    ap.add_argument('--out-nli',  required=True, help='Output NLI CSV path')
    ap.add_argument('--out-bert', required=True, help='Output BERTScore CSV path')
    args = ap.parse_args()

    n_nli  = SCENARIO_COUNTS[args.scenario]['nli']
    n_bert = SCENARIO_COUNTS[args.scenario]['bert']
    expected_nli_pairs  = [f'P{i:02d}'     for i in range(1, n_nli + 1)]
    expected_bert_pairs = [f'BVAL-{i:03d}' for i in range(1, n_bert + 1)]

    print(f'Scenario: {args.scenario} ({n_nli} NLI + {n_bert} BERTScore)')

    raw_dir = Path(args.raw_dir)

    # NLI
    nli_rows = []
    for judge in JUDGES:
        fname = raw_dir / f'nli_{judge}.txt'
        if not fname.exists():
            sys.exit(f'Missing input file: {fname}')
        text = fname.read_text()
        scores = parse_one(text, NLI_PATTERN, expected_nli_pairs, judge, 'NLI')
        for pid in expected_nli_pairs:
            nli_rows.append({'pair_id': pid, 'judge_id': judge, 'score': int(scores[pid])})

    # BERTScore
    bert_rows = []
    for judge in JUDGES:
        fname = raw_dir / f'bertscore_{judge}.txt'
        if not fname.exists():
            sys.exit(f'Missing input file: {fname}')
        text = fname.read_text()
        cats = parse_one(text, BERT_PATTERN, expected_bert_pairs, judge, 'BERTScore')
        for pid in expected_bert_pairs:
            bert_rows.append({'pair_id': pid, 'judge_id': judge, 'category': cats[pid]})

    # Write outputs
    Path(args.out_nli).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_nli, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pair_id', 'judge_id', 'score'])
        w.writeheader(); w.writerows(nli_rows)
    with open(args.out_bert, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['pair_id', 'judge_id', 'category'])
        w.writeheader(); w.writerows(bert_rows)

    print(f'OK NLI:       {len(nli_rows)} rows  ->  {args.out_nli}')
    print(f'OK BERTScore: {len(bert_rows)} rows  ->  {args.out_bert}')


if __name__ == '__main__':
    main()
