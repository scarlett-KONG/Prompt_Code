import json
import math
from pathlib import Path


# Source data
PROBLEMS_PATH = Path("/Users/scarlett/Code/PromptPG-Bryanchlela/PromptPG/data/tabmwp/problems_test.json")
RESULTS_PATH = Path("/Users/scarlett/Code/PromptPG-Bryanchlela/PromptPG/results/tapex/exp7_tapex-base.json")
OUTPUT_PATH = RESULTS_PATH  # overwrite in place; change if you want a copy


def normalize_text(val: str) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    # numeric normalization
    try:
        f = float(s.replace(",", ""))
        # normalize -0.0 to 0
        if math.isclose(f, 0.0, abs_tol=1e-12):
            f = 0.0
        # trim trailing zeros
        s_clean = f"{f:.12g}"
        return s_clean
    except Exception:
        return s.lower()


def equal_norm(a: str, b: str) -> bool:
    try:
        fa = float(a)
        fb = float(b)
        return math.isclose(fa, fb, rel_tol=1e-6, abs_tol=1e-6)
    except Exception:
        return a == b


def main():
    problems = json.loads(PROBLEMS_PATH.read_text())
    data = json.loads(RESULTS_PATH.read_text())

    results = data.get("results", {})
    fixed = {}

    for pid, entry in results.items():
        # Ground-truth answer from problems
        gt_answer = problems.get(pid, {}).get("answer", "")
        gt_answer_norm = normalize_text(gt_answer)

        # Original file fields: "output" actually holds steps; "answer" holds model prediction
        steps = entry.get("output", "")
        prediction = entry.get("answer", "")
        prediction_norm = normalize_text(prediction)

        true_false = equal_norm(gt_answer_norm, prediction_norm)

        fixed[pid] = {
            "answer": gt_answer,
            "answer_norm": gt_answer_norm,
            "steps": steps,
            "output": prediction,  # final predicted answer
            "prediction": prediction,
            "prediction_norm": prediction_norm,
            "true_false": true_false,
        }

    data["results"] = fixed
    OUTPUT_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(f"Updated results written to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

