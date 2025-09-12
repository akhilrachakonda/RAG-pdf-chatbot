import csv
import argparse
from typing import List, Dict, Any
from retriever import ChromaRetriever

def precision_at_k(pred: List[Dict[str, Any]], gold_sources: List[str], k: int) -> float:
    pred_sources = [ (d.get("source") or "").lower() for d in pred[:k] ]
    gold = set([s.lower() for s in gold_sources if s])
    if not gold:
        return 0.0
    hits = sum(1 for s in pred_sources if s in gold)
    return hits / min(k, len(pred_sources) or 1)

def mrr_at_k(pred: List[Dict[str, Any]], gold_sources: List[str], k: int) -> float:
    gold = set([s.lower() for s in gold_sources if s])
    for i, d in enumerate(pred[:k], 1):
        if (d.get("source") or "").lower() in gold:
            return 1.0 / i
    return 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to eval/questions.csv")
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()

    r = ChromaRetriever()
    total = 0
    p_at_k_sum = 0.0
    mrr_sum = 0.0

    rows = []
    with open(args.csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    for row in rows:
        q = row["question"].strip()
        gold_sources = [s.strip() for s in row["expected_sources"].split("|") if s.strip()]
        docs = r.query(q, top_k=args.k)
        p_at_k = precision_at_k(docs, gold_sources, args.k)
        mrr = mrr_at_k(docs, gold_sources, args.k)
        p_at_k_sum += p_at_k
        mrr_sum += mrr
        total += 1
        print(f"Q{total}: P@{args.k}={p_at_k:.2f}  MRR@{args.k}={mrr:.2f}  | {q}")

    if total:
        print("-" * 60)
        print(f"Avg P@{args.k}: {p_at_k_sum/total:.3f}")
        print(f"Avg MRR@{args.k}: {mrr_sum/total:.3f}")
    else:
        print("No rows to evaluate.")

if __name__ == "__main__":
    main()

