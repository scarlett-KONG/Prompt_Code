"""
BEST PRACTICES Tapex Training for TabMWP

Features:
1) Table-tuning: synthetic table QA tasks
2) Chain-of-thought: train with rationale + answer
3) Structured table formatting: natural language rows/columns
4) Curriculum learning: easy → hard ordering
5) Optional eval export: writes results with answer/answer_norm/steps/output/prediction/prediction_norm/true_false

Usage:
  # Full training with all improvements
  python train_best.py \
      --data_root ./data/tabmwp \
      --model tapex-large \
      --use_table_tuning \
      --use_chain_of_thought \
      --use_structured_input \
      --use_curriculum \
      --epochs 15 \
      --gpu 0

  # Quick test + save eval-style predictions
  python train_best.py \
      --data_root ./data/tabmwp \
      --model tapex-base \
      --use_table_tuning \
      --augment_ratio 0.3 \
      --epochs 5 \
      --gpu 0 \
      --save_predictions \
      --eval_split dev

Quick run (classic train/infer/eval, same as Run_experiment_Tapex_quick.ipynb):
  cd run_tapex
  python train.py --label exp2 \\
      --model tapex-base \\
      --batch_size 16 \\
      --eval_batch_size 16 \\
      --gpu 0 \\
      --save_all

  python inference.py --label exp2 \\
      --test_split test \\
      --test_num -1 \\
      --model tapex-base \\
      --gpu 0 \\
      --check_point best_model.pth

  python eval.py --result_file tapex/exp2_tapex-base.json
"""

import os
import sys
import json
import random
import argparse
from typing import List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,
)

# Ensure project root is on path for `tools`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import utils

# ============================================================
# MODEL CHOICES
# ============================================================

tapex_models = {
    "tapex-base": "microsoft/tapex-base-finetuned-wtq",
    "tapex-large": "microsoft/tapex-large-finetuned-wtq",
}

bart_models = {
    "tapex-base": "facebook/bart-base",
    "tapex-large": "facebook/bart-large",
}


# ============================================================
# TABLE TASK GENERATOR (same as T5 version)
# ============================================================


class EnhancedTableTaskGenerator:
    """Generate diverse synthetic table tasks."""

    def parse_table(self, table_str: str) -> dict:
        """Parse TabMWP table string."""
        try:
            lines = table_str.strip().split("\n")
            headers = [h.strip() for h in lines[0].split("|")]

            rows = []
            for line in lines[1:]:
                if line.strip():
                    row = [cell.strip() for cell in line.split("|")]
                    while len(row) < len(headers):
                        row.append("")
                    row = row[: len(headers)]
                    rows.append(row)

            return {"headers": headers, "rows": rows}
        except Exception:
            return None

    def get_numeric_values(self, parsed: dict, col_idx: int) -> list:
        """Extract numeric values from a column."""
        values = []
        for row in parsed["rows"]:
            try:
                val = (
                    row[col_idx]
                    .replace("$", "")
                    .replace(",", "")
                    .replace("%", "")
                )
                values.append(float(val))
            except Exception:
                pass
        return values

    def generate_tasks(
        self, table_str: str, table_title: str = "", num_tasks: int = 3
    ) -> list:
        """Generate multiple synthetic tasks for one table."""
        parsed = self.parse_table(table_str)
        if not parsed or len(parsed["rows"]) == 0:
            return []

        tasks = []
        task_generators = [
            self._cell_lookup,
            self._column_sum,
            self._column_max,
            self._column_min,
            self._column_average,
            self._row_count,
            self._column_count,
            self._compare_values,
            self._difference_task,
            self._which_more,
            self._total_task,
        ]

        if table_title:
            _ = f"{table_title}\n{table_str}"

        random.shuffle(task_generators)

        for gen in task_generators[:num_tasks]:
            try:
                task = gen(parsed)
                if task:
                    tasks.append(task)
            except Exception:
                pass

        return tasks

    def _cell_lookup(self, parsed: dict) -> dict:
        """Simple cell lookup task."""
        row_idx = random.randint(0, len(parsed["rows"]) - 1)
        col_idx = random.randint(0, len(parsed["headers"]) - 1)

        row_key = parsed["rows"][row_idx][0]
        col_name = parsed["headers"][col_idx]
        value = parsed["rows"][row_idx][col_idx]

        question = f"Look at the table. What is the {col_name} for {row_key}?"

        return {"question": question, "answer": value, "task": "cell_lookup"}

    def _column_sum(self, parsed: dict) -> dict:
        """Sum a numeric column."""
        for col_idx, header in enumerate(parsed["headers"]):
            values = self.get_numeric_values(parsed, col_idx)
            if len(values) == len(parsed["rows"]) and len(values) > 1:
                total = sum(values)
                answer = str(int(total)) if total == int(total) else f"{total:.2f}"
                question = f"What is the total of all {header} values?"
                return {"question": question, "answer": answer, "task": "column_sum"}
        return None

    def _column_max(self, parsed: dict) -> dict:
        """Find maximum in a column."""
        for col_idx, header in enumerate(parsed["headers"]):
            values = self.get_numeric_values(parsed, col_idx)
            if len(values) >= 2:
                max_val = max(values)
                answer = str(int(max_val)) if max_val == int(max_val) else f"{max_val:.2f}"
                question = f"What is the highest {header}?"
                return {"question": question, "answer": answer, "task": "column_max"}
        return None

    def _column_min(self, parsed: dict) -> dict:
        """Find minimum in a column."""
        for col_idx, header in enumerate(parsed["headers"]):
            values = self.get_numeric_values(parsed, col_idx)
            if len(values) >= 2:
                min_val = min(values)
                answer = str(int(min_val)) if min_val == int(min_val) else f"{min_val:.2f}"
                question = f"What is the lowest {header}?"
                return {"question": question, "answer": answer, "task": "column_min"}
        return None

    def _column_average(self, parsed: dict) -> dict:
        """Calculate average of a column."""
        for col_idx, header in enumerate(parsed["headers"]):
            values = self.get_numeric_values(parsed, col_idx)
            if len(values) >= 2:
                avg = sum(values) / len(values)
                answer = f"{avg:.2f}"
                question = f"What is the mean {header}?"
                return {"question": question, "answer": answer, "task": "column_average"}
        return None

    def _row_count(self, parsed: dict) -> dict:
        """Count rows."""
        question = "How many rows are in this table?"
        answer = str(len(parsed["rows"]))
        return {"question": question, "answer": answer, "task": "row_count"}

    def _column_count(self, parsed: dict) -> dict:
        """Count columns."""
        question = "How many columns are in this table?"
        answer = str(len(parsed["headers"]))
        return {"question": question, "answer": answer, "task": "column_count"}

    def _compare_values(self, parsed: dict) -> dict:
        """Compare two values."""
        if len(parsed["rows"]) < 2:
            return None

        for col_idx, header in enumerate(parsed["headers"]):
            values = self.get_numeric_values(parsed, col_idx)
            if len(values) >= 2:
                idx1, idx2 = random.sample(range(len(parsed["rows"])), 2)
                row1_key = parsed["rows"][idx1][0]
                row2_key = parsed["rows"][idx2][0]

                val1 = values[idx1] if idx1 < len(values) else None
                val2 = values[idx2] if idx2 < len(values) else None

                if val1 is not None and val2 is not None:
                    if val1 > val2:
                        answer = row1_key
                    elif val2 > val1:
                        answer = row2_key
                    else:
                        answer = "same"

                    question = f"Which has more {header}, {row1_key} or {row2_key}?"
                    return {"question": question, "answer": answer, "task": "compare"}
        return None

    def _difference_task(self, parsed: dict) -> dict:
        """Calculate difference between two values."""
        if len(parsed["rows"]) < 2:
            return None

        for col_idx, header in enumerate(parsed["headers"]):
            values = self.get_numeric_values(parsed, col_idx)
            if len(values) >= 2:
                idx1, idx2 = random.sample(range(len(values)), 2)
                row1_key = parsed["rows"][idx1][0]
                row2_key = parsed["rows"][idx2][0]

                diff = abs(values[idx1] - values[idx2])
                answer = str(int(diff)) if diff == int(diff) else f"{diff:.2f}"

                if values[idx1] > values[idx2]:
                    question = f"How many more {header} does {row1_key} have than {row2_key}?"
                else:
                    question = f"How many more {header} does {row2_key} have than {row1_key}?"

                return {"question": question, "answer": answer, "task": "difference"}
        return None

    def _which_more(self, parsed: dict) -> dict:
        """Which row has more of something."""
        for col_idx, header in enumerate(parsed["headers"]):
            values = self.get_numeric_values(parsed, col_idx)
            if len(values) == len(parsed["rows"]) and len(values) >= 2:
                max_idx = values.index(max(values))
                answer = parsed["rows"][max_idx][0]
                question = f"Which has the most {header}?"
                return {"question": question, "answer": answer, "task": "which_more"}
        return None

    def _total_task(self, parsed: dict) -> dict:
        """Sum specific rows."""
        if len(parsed["rows"]) < 2:
            return None

        for col_idx, header in enumerate(parsed["headers"]):
            values = self.get_numeric_values(parsed, col_idx)
            if len(values) == len(parsed["rows"]):
                num_rows = min(random.randint(2, 3), len(parsed["rows"]))
                indices = random.sample(range(len(parsed["rows"])), num_rows)

                row_names = [parsed["rows"][i][0] for i in indices]
                total = sum(values[i] for i in indices)

                answer = str(int(total)) if total == int(total) else f"{total:.2f}"
                question = f"What is the total {header} for {' and '.join(row_names)}?"

                return {"question": question, "answer": answer, "task": "total_specific"}
        return None


# ============================================================
# STRUCTURED TABLE FORMAT
# ============================================================


def convert_to_structured_format(table_str: str, table_title: str = "") -> str:
    """
    Convert table to structured format that's easier for model to understand.
    """
    try:
        lines = table_str.strip().split("\n")
        headers = [h.strip() for h in lines[0].split("|")]

        result = []
        if table_title:
            result.append(f"Table: {table_title}")
        else:
            result.append(f"Table with columns: {', '.join(headers)}")

        for i, line in enumerate(lines[1:]):
            if line.strip():
                values = [v.strip() for v in line.split("|")]
                pairs = []
                for h, v in zip(headers, values):
                    pairs.append(f"{h} = {v}")
                result.append(f"Row {i+1}: {', '.join(pairs)}")

        return "\n".join(result)
    except Exception:
        return table_str


# ============================================================
# CHAIN OF THOUGHT
# ============================================================


def create_cot_output(problem: dict) -> str:
    solution = problem.get("solution", "")
    answer = str(problem["answer"])

    if solution:
        solution = solution.strip()
        return f"{solution}\nAnswer: {answer}"
    else:
        return answer


# ============================================================
# CURRICULUM LEARNING
# ============================================================


def sort_by_difficulty(problems: dict) -> list:
    difficulty_scores = []

    for pid, problem in problems.items():
        score = 0

        grade = problem.get("grade", 5)
        score += grade * 10

        if problem.get("ques_type") == "free_text":
            score += 20

        ans_type = problem.get("ans_type", "")
        if ans_type == "decimal_number":
            score += 30
        elif ans_type == "integer_number":
            score += 10

        solution = problem.get("solution", "")
        score += len(solution) // 50

        difficulty_scores.append((pid, problem, score))

    difficulty_scores.sort(key=lambda x: x[2])

    return [(pid, problem) for pid, problem, _ in difficulty_scores]


# ============================================================
# NORMALIZATION / PRED EXTRACTION (from tapex eval)
# ============================================================


def replace_punctuation(text: str) -> str:
    return text.replace('"', "").replace("'", "")


def fix_buggy_characters(text: str) -> str:
    import re

    return re.sub(r"[{}^\\`\u2047<]", " ", text)


def score_string_similarity(str1: str, str2: str) -> float:
    import re
    import numpy as np

    if str1 == str2:
        return 3.0
    str1 = fix_buggy_characters(replace_punctuation(str1))
    str2 = fix_buggy_characters(replace_punctuation(str2))
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0


def extract_prediction(output: str, options: List[str]):
    import numpy as np
    import re

    if options:
        scores = [score_string_similarity(x, output) for x in options]
        max_idx = int(np.argmax(scores))
        return options[max_idx]

    patterns = [
        r" ([\d\$\.\,\/\:]+ [AP]\.M\.)",
        r"([\-\d\$\.\,\/\:]{0,}[\d]+)",
    ]

    for p in patterns:
        pattern = re.compile(p)
        res = pattern.findall(output)
        if len(res) > 0:
            return res[-1].strip()

    return output


def normalize_answer(text: str, unit: str) -> str:
    import re

    text = re.sub("^[\$]", "", text)
    text = re.sub("[\\,\\.\\,\\/]$", "", text)

    result = re.match("^[-+]?[\d,./]+$", text)

    if result is not None:
        text = text.replace(",", "")
        result = re.match("[-+]?\\d+$", text)

        if result is not None:
            number = int(text)
        elif "/" in text:
            nums = text.split("/")
            number = round(float(nums[0]) / float(nums[1]), 3)
        else:
            number = round(float(text), 3)
        number = str(number)
        number = re.sub(r"\.[0]+$", "", number)
        return number
    else:
        if unit:
            text = text.replace(unit, "").strip()
        return text


# ============================================================
# DATASET
# ============================================================


class TapexBestPracticesDataset(Dataset):
    """Dataset with table-tuning, CoT, structured input, curriculum."""

    def __init__(self, problems: dict, args, is_train: bool = True):
        self.max_input_length = args.max_input_length
        self.max_output_length = args.max_output_length
        self.use_structured = args.use_structured_input
        self.use_cot = args.use_chain_of_thought

        option_inds = ["A", "B", "C", "D", "E", "F"]

        if args.use_curriculum and is_train:
            problem_list = sort_by_difficulty(problems)
        else:
            problem_list = list(problems.items())

        self.examples: List[Dict] = []

        for pid, problem in problem_list:
            table_str = problem["table"]
            table_title = problem.get("table_title", "")

            if self.use_structured:
                table_text = convert_to_structured_format(table_str, table_title)
            else:
                if table_title:
                    table_text = f"{table_title}\n{table_str}"
                else:
                    table_text = table_str

            question = problem["question"]
            unit = problem.get("unit", "")
            choices = problem.get("choices", [])

            if unit:
                question += f" (Unit: {unit})"
            if choices:
                for i, c in enumerate(choices):
                    question += f" ({option_inds[i]}) {c}"

            input_text = f"{table_text}\n{question}"
            input_text = input_text.replace("\n", " \\n ")

            if self.use_cot:
                output_text = create_cot_output(problem)
            else:
                output_text = str(problem["answer"])

            self.examples.append(
                {
                    "pid": pid,
                    "table_pd": utils.convert_table_text_to_pandas(table_str),
                    "input_text": input_text,
                    "output_text": output_text,
                    "answer": str(problem["answer"]),
                    "choices": choices,
                    "unit": unit,
                    "steps": problem.get("solution", ""),
                }
            )

        if args.use_table_tuning and is_train:
            generator = EnhancedTableTaskGenerator()
            num_synthetic = int(len(self.examples) * args.augment_ratio)

            print(f"Generating {num_synthetic} synthetic table tasks...")

            synthetic_added = 0
            for pid, problem in problems.items():
                if synthetic_added >= num_synthetic:
                    break

                table_str = problem["table"]
                table_title = problem.get("table_title", "")

                tasks = generator.generate_tasks(table_str, table_title, num_tasks=2)

                for task in tasks:
                    if synthetic_added >= num_synthetic:
                        break

                    if self.use_structured:
                        table_text = convert_to_structured_format(table_str, table_title)
                    else:
                        if table_title:
                            table_text = f"{table_title}\n{table_str}"
                        else:
                            table_text = table_str

                    input_text = f"{table_text}\n{task['question']}"
                    input_text = input_text.replace("\n", " \\n ")

                    self.examples.append(
                        {
                            "pid": f"synthetic_{synthetic_added}",
                            "table_pd": utils.convert_table_text_to_pandas(table_str),
                            "input_text": input_text,
                            "output_text": task["answer"],
                            "answer": task["answer"],
                            "choices": [],
                            "unit": "",
                            "steps": "",
                        }
                    )
                    synthetic_added += 1

            print(f"Added {synthetic_added} synthetic tasks")

        if is_train and not args.use_curriculum:
            random.shuffle(self.examples)

        print(f"Total examples: {len(self.examples)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def collate_fn(batch):
    return {
        "pids": [b["pid"] for b in batch],
        "tables": [b["table_pd"] for b in batch],
        "inputs": [b["input_text"] for b in batch],
        "outputs": [b["output_text"] for b in batch],
        "answers": [b["answer"] for b in batch],
        "choices": [b["choices"] for b in batch],
        "units": [b["unit"] for b in batch],
        "steps": [b["steps"] for b in batch],
    }


# ============================================================
# TRAINING / EVAL
# ============================================================


def evaluate_loss(model, dataloader, tokenizer, bart_tokenizer, device, args):
    model.eval()
    eval_loss = 0

    with torch.no_grad():
        for batch in iter(dataloader):
            input_ids = tokenizer(
                batch["tables"],
                batch["inputs"],
                padding="longest",
                truncation=True,
                max_length=args.max_input_length,
                return_tensors="pt",
            ).input_ids.to(device)

            output_ids = (
                bart_tokenizer(
                    batch["outputs"],
                    padding="longest",
                    truncation=True,
                    max_length=args.max_output_length,
                    return_tensors="pt",
                )
                .input_ids.to(device)
            )

            outputs = model(input_ids=input_ids, labels=output_ids)
            loss = outputs.loss
            eval_loss += loss.item() * outputs.logits.shape[0]

    eval_loss /= len(dataloader.dataset)
    model.train()
    return eval_loss


def run_prediction(
    model,
    tokenizer,
    dataloader,
    device,
    args,
):
    model.eval()
    results = {}

    with torch.no_grad():
        for batch in iter(dataloader):
            model_inputs = tokenizer(
                batch["tables"],
                batch["inputs"],
                padding="longest",
                truncation=True,
                max_length=args.max_input_length,
                return_tensors="pt",
            ).to(device)

            outputs = model.generate(
                **model_inputs, max_length=args.max_output_length
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for pid, answer, unit, choices, steps, output in zip(
                batch["pids"],
                batch["answers"],
                batch["units"],
                batch["choices"],
                batch["steps"],
                decoded,
            ):
                prediction = extract_prediction(output, choices)
                answer_norm = normalize_answer(str(answer), unit)
                prediction_norm = normalize_answer(str(prediction), unit)
                is_correct = (
                    str(answer_norm).lower() == str(prediction_norm).lower()
                )

                results[pid] = {
                    "answer": str(answer),
                    "answer_norm": answer_norm,
                    "steps": steps,
                    "output": output,
                    "prediction": prediction,
                    "prediction_norm": prediction_norm,
                    "true_false": is_correct,
                }

    return results


def train(args):
    print("=" * 70)
    print("BEST PRACTICES TAPEX TRAINING FOR TABMWP")
    print("=" * 70)
    print(f"Techniques enabled:")
    print(f"  - Table-tuning: {args.use_table_tuning} (ratio: {args.augment_ratio})")
    print(f"  - Chain-of-thought: {args.use_chain_of_thought}")
    print(f"  - Structured input: {args.use_structured_input}")
    print(f"  - Curriculum learning: {args.use_curriculum}")
    print()

    model_name = tapex_models[args.model]
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    bart_tokenizer = AutoTokenizer.from_pretrained(bart_models[args.model])

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    model.to(device)

    train_file = os.path.join(args.data_root, "problems_train.json")
    val_file = os.path.join(args.data_root, "problems_dev.json")

    print(f"\nLoading data...")
    if args.train_num and args.train_num >0:
        train_problems = json.load(open(train_file))
        keys = list(train_problems.keys())[: args.train_num]
        train_problems = {k: train_problems[k] for k in keys}
        print(f"Using first {len(train_problems)} train examples")
    else:
        train_problems = json.load(open(train_file))

    val_problems = json.load(open(val_file))

    print(f"\nCreating training dataset...")
    train_dataset = TapexBestPracticesDataset(
        train_problems, args, is_train=True
    )

    print(f"\nCreating validation dataset...")
    val_dataset = TapexBestPracticesDataset(
        val_problems, args, is_train=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not args.use_curriculum,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, total_steps // 10)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps
    )

    best_val_loss = float("inf")
    output_dir = os.path.join(args.output, args.label)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nStarting training for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0

        for batch in iter(train_loader):
            input_ids = tokenizer(
                batch["tables"],
                batch["inputs"],
                padding="longest",
                truncation=True,
                max_length=args.max_input_length,
                return_tensors="pt",
            ).input_ids.to(device)

            output_ids = (
                bart_tokenizer(
                    batch["outputs"],
                    padding="longest",
                    truncation=True,
                    max_length=args.max_output_length,
                    return_tensors="pt",
                )
                .input_ids.to(device)
            )

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=output_ids)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * outputs.logits.shape[0]

        avg_train_loss = train_loss / len(train_loader.dataset)

        val_loss = evaluate_loss(
            model, val_loader, tokenizer, bart_tokenizer, device, args
        )

        print(
            f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model (val_loss: {best_val_loss:.4f})")

    print(f"\n{'='*70}")
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}/best_model.pth")
    print(f"{'='*70}")

    if args.save_predictions:
        eval_split = args.eval_split
        data_file = os.path.join(args.data_root, f"problems_{eval_split}.json")
        problems = json.load(open(data_file))
        eval_dataset = TapexBestPracticesDataset(
            problems, args, is_train=False
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        print(f"\nRunning prediction export on split: {eval_split}")
        results = run_prediction(
            model, tokenizer, eval_loader, device, args
        )

        correct = sum(1 for r in results.values() if r["true_false"])
        total = len(results)
        acc = correct / total * 100

        out_dir = args.result_root
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"{args.label}_{args.model}_{eval_split}.json"
        )

        payload = {
            "acc": acc,
            "correct": correct,
            "count": total,
            "args": vars(args),
            "results": results,
        }

        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, separators=(",", ": "))

        print(
            f"Saved prediction file to {out_path} "
            f"(fields: answer, answer_norm, steps, output, prediction, prediction_norm, true_false)"
        )


def parse_args():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument("--data_root", type=str, default="../data/tabmwp")
    parser.add_argument("--output", type=str, default="../saved_models/tapex_best")
    parser.add_argument("--result_root", type=str, default="../results/tapex_best")
    parser.add_argument("--train_num", type=int, default=-1,help="Use only first N train examples; -1 = all")
    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="tapex-base",
        choices=["tapex-base", "tapex-large"],
    )
    parser.add_argument("--label", type=str, default="exp_best")

    # Techniques
    parser.add_argument(
        "--use_table_tuning",
        action="store_true",
        help="Add synthetic table understanding tasks",
    )
    parser.add_argument(
        "--use_chain_of_thought",
        action="store_true",
        help="Train with step-by-step solutions",
    )
    parser.add_argument(
        "--use_structured_input",
        action="store_true",
        help="Use structured table format",
    )
    parser.add_argument(
        "--use_curriculum",
        action="store_true",
        help="Train easy→hard (curriculum learning)",
    )

    # Augmentation
    parser.add_argument(
        "--augment_ratio",
        type=float,
        default=0.3,
        help="Ratio of synthetic tasks (for table-tuning)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--gpu", type=str, default="0")

    # Tokenization
    parser.add_argument("--max_input_length", type=int, default=512)
    parser.add_argument("--max_output_length", type=int, default=150)

    # Prediction export
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="After training, run eval split and save predictions JSON",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["dev", "dev1k", "test", "test1k"],
    )

    args = parser.parse_args()

    print("====Arguments====")
    print(json.dumps(vars(args), indent=2))
    print()

    return args


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()

