# pip install transformers datasets accelerate
import os, tempfile, subprocess, argparse, json, sys, shutil
from typing import Optional
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

PROMPT_TMPL = (
    "You are a helpful Java coding assistant.\n"
    "Task: {task}\n"
    "Signature: {sig}\n"
    "Return ONLY Java code.\n"
)

def make_prompt(rec):
    task = rec.get("human_label") or rec.get("docstring") or ""
    sig  = rec.get("name") or ""
    return PROMPT_TMPL.format(task=str(task).strip(), sig=str(sig).strip())

def extract_java(text: str) -> str:
    # keep only the Java code; strip fences if present
    t = text
    if "Return ONLY Java code." in t:
        t = t.split("Return ONLY Java code.", 1)[-1].strip()
    if "```java" in t:
        t = t.split("```java")[-1].split("```")[0].strip()
    elif "```" in t:
        t = t.split("```")[-1].split("```")[0].strip()
    return t.strip()

def javac_compile(java_code: str, classname: str = "Solution") -> bool:
    """Try to compile the code. If it lacks a class, wrap into a minimal class."""
    with tempfile.TemporaryDirectory() as td:
        code = java_code
        # If no 'class' keyword found, wrap in a class
        if " class " not in code and not code.strip().startswith("class "):
            # Try to detect if it looks like a single method; just wrap it
            code = f"public class {classname} {{\n{java_code}\n}}\n"
        src = os.path.join(td, f"{classname}.java") if "class " not in java_code else os.path.join(td, "Main.java")
        with open(src, "w", encoding="utf-8") as f:
            f.write(code)
        try:
            subprocess.check_output(["javac", src], stderr=subprocess.STDOUT)
            return True
        except subprocess.CalledProcessError as e:
            # Print first lines of error for debugging
            msg = e.output.decode(errors="ignore")
            sys.stderr.write("[javac error] " + "\n".join(msg.splitlines()[:5]) + "\n")
            return False

def run_tests_with_command(java_code: str, test_cmd: str, classname: str = "Solution") -> bool:
    """
    Optional test runner. We create a temp project dir, drop the generated code into ./src/main/java/,
    then run a user-provided command that must return 0 on success.
    You need to ensure your test_cmd knows how to compile and run tests (e.g., Maven).
    """
    with tempfile.TemporaryDirectory() as td:
        # Expected Java project layout (Maven-like)
        src_dir = os.path.join(td, "src", "main", "java")
        os.makedirs(src_dir, exist_ok=True)
        java_path = os.path.join(src_dir, f"{classname}.java")

        code = java_code
        if " class " not in code and not code.strip().startswith("class "):
            code = f"public class {classname} {{\n{java_code}\n}}\n"

        with open(java_path, "w", encoding="utf-8") as f:
            f.write(code)

        # Run user test command inside temp dir
        try:
            result = subprocess.run(test_cmd, cwd=td, shell=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            # Uncomment to debug:
            # print(result.stdout[:500])
            return result.returncode == 0
        except Exception as e:
            sys.stderr.write(f"[test error] {e}\n")
            return False

def generate(model, tok, prompt: str, max_new_tokens: int = 256) -> str:
    inputs = tok(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    return extract_java(text)

def evaluate(model_path: str, base_model_name: Optional[str], dataset_path: str,
             limit: Optional[int], do_tests: bool, test_cmd: Optional[str]) -> dict:
    # Tokenizer from base name (safer) or from model_path if base not provided
    tok_src = base_model_name or model_path
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True)

    ds = load_dataset("json", data_files=dataset_path, field="RECORDS")["train"]
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    total = len(ds)
    comp_ok = 0
    test_ok = 0

    for i, rec in enumerate(ds):
        prompt = make_prompt(rec)
        code = generate(model, tok, prompt)
        if javac_compile(code):
            comp_ok += 1
            if do_tests and test_cmd:
                if run_tests_with_command(code, test_cmd):
                    test_ok += 1

        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{total}] compile_ok={comp_ok}, tests_ok={test_ok}")

    metrics = {
        "samples": total,
        "compile_rate": comp_ok / total if total else 0.0,
        "pass_at_1"   : (test_ok / total if total else 0.0) if do_tests else None
    }
    return metrics

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="./CoderEval4Java.json")
    ap.add_argument("--base_model", type=str, required=True, help="e.g. Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--trained_model", type=str, required=True, help="path to your GRPO/PPO fine-tuned model")
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--run_tests", action="store_true", help="run unit tests via --test_cmd if provided")
    ap.add_argument("--test_cmd", type=str, default=None, help="shell command that compiles & runs tests; returns 0 on pass")
    args = ap.parse_args()

    print("Evaluating BASE model...")
    base_metrics = evaluate(args.base_model, args.base_model, args.dataset, args.limit, args.run_tests, args.test_cmd)
    print("BASE metrics:", base_metrics)

    print("\nEvaluating TRAINED model...")
    trained_metrics = evaluate(args.trained_model, args.base_model, args.dataset, args.limit, args.run_tests, args.test_cmd)
    print("TRAINED metrics:", trained_metrics)

    print("\nΔ compile_rate:", (trained_metrics["compile_rate"] - base_metrics["compile_rate"]))
    if args.run_tests and base_metrics["pass_at_1"] is not None and trained_metrics["pass_at_1"] is not None:
        print("Δ pass@1:", (trained_metrics["pass_at_1"] - base_metrics["pass_at_1"]))

if __name__ == "__main__":
    main()
