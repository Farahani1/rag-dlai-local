"""Choose the embedding router's threshold on the dev set (not on the benchmark).

    python benchmark/tune_router.py            # show scores and the best threshold
    python benchmark/tune_router.py --write    # also store it in router_dev.yaml

Each dev query is scored by its highest cosine similarity to the FAQ questions
in data/faq.yaml, using the same embedding model the W5 pipeline uses. The
threshold with the best dev accuracy (widest margin on ties) is chosen.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "w5")):
    if p not in sys.path:
        sys.path.insert(0, p)

from benchmark.variants import ROUTER_DEV_PATH, best_threshold, max_similarity  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true", help="Store the threshold in router_dev.yaml")
    args = parser.parse_args(argv)

    from embedding import embed_query  # w5/embedding.py, as used by the notebook
    from setting import config

    dev = yaml.safe_load(ROUTER_DEV_PATH.read_text(encoding="utf-8"))
    faq = joblib.load(str(config.faqData))
    matrix = np.asarray([embed_query(item["question"]) for item in faq], dtype=np.float32)

    scores = {label: [max_similarity(embed_query(q), matrix) for q in dev[label]] for label in ("faq", "product")}
    threshold, accuracy = best_threshold(scores["faq"], scores["product"])

    for label in ("faq", "product"):
        print(f"{label}:")
        for q, s in sorted(zip(dev[label], scores[label]), key=lambda x: x[1]):
            print(f"  {s:.3f}  {q}")
    n = len(scores["faq"]) + len(scores["product"])
    print(f"\nthreshold = {threshold}  (dev accuracy {accuracy:.0%} on {n} queries)")

    if args.write:
        text = ROUTER_DEV_PATH.read_text(encoding="utf-8")
        text, count = re.subn(r"^threshold: .*$", f"threshold: {threshold}", text, count=1, flags=re.M)
        if count != 1:
            raise SystemExit("No 'threshold:' line in router_dev.yaml")
        ROUTER_DEV_PATH.write_text(text, encoding="utf-8")
        print(f"Wrote threshold to {ROUTER_DEV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
