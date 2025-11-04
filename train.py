"""
PPO training on CoderEval4Java.json using ONLY static metrics (no LLM judge).

Usage:
  python train_codereval_ppo_static.py \
    --data_root /data_root/codereval_java \
    --model Qwen2.5-Coder-7B-Instruct \
    --out_dir ./runs/codereval_static \
    --batch_size 32 --mini_batch_size 8 --ppo_epochs 3

Deps:
  pip install "transformers>=4.43" "accelerate>=0.30" peft bitsandbytes datasets trl==0.8.6
"""

import os, re, json, argparse, torch
from dataclasses import dataclass
from typing import List, Optional, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import PPOConfig, PPOTrainer
from datasets import Dataset
# ----------------------------
# Static metric utilities
# ----------------------------

DECISION_TOKENS = ("if", "for", "while", "case", "catch", "&&", "||", "?")

def cyclomatic_complexity(code: str) -> int:
    count = 1
    for tok in DECISION_TOKENS:
        count += len(re.findall(r"\b" + re.escape(tok) + r"\b", code))
    return max(count, 1)

def lines_of_code(code: str) -> int:
    lines = [l for l in code.splitlines() if l.strip() and not l.strip().startswith("//")]
    return len(lines)

def max_nesting_depth(code: str) -> int:
    depth, max_depth = 0, 0
    tokens = re.findall(r"[{}]|\b(if|for|while|switch|try|catch)\b", code)
    for t in tokens:
        if t == "{":
            depth += 1; max_depth = max(max_depth, depth)
        elif t == "}":
            depth -= 1
    return max_depth

def comment_ratio(code: str) -> float:
    total = len(code.splitlines())
    if total == 0: return 0.0
    comment = sum(1 for l in code.splitlines()
                  if l.strip().startswith("//") or "/*" in l or "*/" in l)
    return min(max(comment / total, 0.0), 1.0)

def cbo_estimate(code: str) -> int:
    # crude coupling proxy: distinct Type-like tokens
    candidates = set(re.findall(r"\b([A-Z][A-Za-z0-9_]{2,})\b", code))
    blacklist = {"Class","String","Integer","Long","Double","Boolean","List","Map","Set","System","Object"}
    return max(len(candidates - blacklist), 0)

def duplication_ratio(code: str, window: int = 6) -> float:
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[{}();,.=+\-*/<>!&|?:]", code)
    if len(toks) < window*2: return 0.0
    grams = {}
    for i in range(len(toks)-window+1):
        key = tuple(toks[i:i+window])
        grams[key] = grams.get(key, 0)+1
    dup_tokens = sum((c-1)*window for c in grams.values() if c>1)
    return min(dup_tokens / max(len(toks),1), 1.0)

def normalize(x, lo, hi, invert=False):
    if hi <= lo: return 0.0
    y = (x - lo) / (hi - lo)
    y = min(max(y, 0.0), 1.0)
    return 1.0 - y if invert else y

def metric_bundle(code: str) -> Dict[str, float]:
    cc = cyclomatic_complexity(code)
    loc = lines_of_code(code)
    depth = max_nesting_depth(code)
    comm = comment_ratio(code)
    cbo = cbo_estimate(code)
    dup = duplication_ratio(code)

    # Empirical ranges (tune with dataset profiling if desired)
    norm = {
        "cc":       normalize(cc, 1, 20,  invert=True),
        "loc":      normalize(loc, 5, 120, invert=True),
        "depth":    normalize(depth, 1, 6, invert=True),
        "comments": normalize(comm, 0.05, 0.35, invert=False),
        "cbo":      normalize(cbo, 0, 25,  invert=True),
        "dup":      normalize(dup, 0.0, 0.25, invert=True),
    }
    complexity  = (norm["cc"] + norm["depth"] + norm["loc"]) / 3.0
    modularity  = (norm["cbo"] + norm["dup"]) / 2.0
    readability = norm["comments"]

    return {
        "cc_raw": cc, "loc_raw": loc, "depth_raw": depth,
        "comm_raw": comm, "cbo_raw": cbo, "dup_raw": dup,
        "complexity": complexity, "modularity": modularity, "readability": readability
    }

# The reward is only from static metrics (no correctness from a judge/tests here)
def compute_reward(static_m: Dict[str, float],
                   weights: Dict[str, float]) -> float:
    R = (weights["complexity"]  * static_m["complexity"] +
         weights["modularity"]  * static_m["modularity"] +
         weights["readability"] * static_m["readability"])
    return float(max(0.0, min(1.0, R)))

# ----------------------------
# Data: CoderEval4Java.json
# ----------------------------
@dataclass
class TaskItem:
    id: str
    prompt: str          # human_label or docstring
    all_context: str
    signature: str       # method/class name if present
    code: str            # reference code (unused for training, handy for analysis)
    docstring: str
    human_label: str
    context: Optional[str] = None

def load_codereval_records(data_root: str, limit: Optional[int]=None) -> List[TaskItem]:
    path = os.path.join(data_root, "CoderEval4Java.json")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    recs = raw.get("RECORDS", [])
    tasks: List[TaskItem] = []
    for r in recs:
        tasks.append(TaskItem(
            id=str(r.get("_id","")),
            prompt=r.get("human_label", r.get("docstring","")),
            all_context=r.get("all_context",""),
            signature=r.get("name",""),
            code=r.get("code",""),
            docstring=r.get("docstring",""),
            human_label=r.get("human_label",""),
            context=r.get("file_content", None)
        ))
        if limit and len(tasks) >= limit:
            break
    return tasks

def build_prompt(t: TaskItem) -> str:
    parts = []
    parts.append("You are a helpful Java coding assistant. Write clean, idiomatic Java.")
    parts.append(f"Task:\n{t.prompt}")
    if t.signature:
        parts.append(f"Target method/class name:\n{t.signature}")
    parts.append("Return ONLY the Java code. No explanations.")
    return "\n\n".join(parts)

# ----------------------------
# PPO trainer (QLoRA)
# ----------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data",
                    help="Folder containing CoderEval4Java.json")
    ap.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--out_dir", type=str, default="./runs/codereval_static")
    ap.add_argument("--limit", type=int, default=None)

    # quant + LoRA
    ap.add_argument("--load_in_4bit", action="store_true", default=True)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)

    # PPO
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--mini_batch_size", type=int, default=8)
    ap.add_argument("--ppo_epochs", type=int, default=3)
    ap.add_argument("--clip_range", type=float, default=0.2)
    ap.add_argument("--kl_target", type=float, default=0.08)

    # generation
    ap.add_argument("--gen_max_new_tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)

    # logging/checkpoints
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=500)

    # reward weights (no correctness; only static metrics)
    ap.add_argument("--w_complexity", type=float, default=0.40)
    ap.add_argument("--w_modularity", type=float, default=0.35)
    ap.add_argument("--w_readability", type=float, default=0.25)

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # data
    tasks = load_codereval_records("./data", args.limit)
    prompts = [build_prompt(t) for t in tasks]
    ds = Dataset.from_dict({"prompt": prompts})

    # model
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    lcfg = LoraConfig(
        r=args.r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lcfg)

    # PPO
    ppo_cfg = PPOConfig(
        model_name=args.model,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        mini_batch_size=args.mini_batch_size,
        ppo_epochs=args.ppo_epochs,
        clip_range=args.clip_range,
        target_kl=args.kl_target,
        remove_unused_columns=False
    )
    trainer = PPOTrainer(ppo_cfg, model, tok)

    gen_kwargs = dict(
        max_new_tokens=args.gen_max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        pad_token_id=tok.eos_token_id
    )
    weights = {
        "complexity":  args.w_complexity,
        "modularity":  args.w_modularity,
        "readability": args.w_readability,
    }

    # training loop
    for start in range(0, len(ds), ppo_cfg.batch_size):
        batch = ds[start:start+ppo_cfg.batch_size]
        if len(batch["prompt"]) == 0:
            break

        # generate
        input_ids = tok(batch["prompt"], return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            gen = model.generate(**input_ids, **gen_kwargs)

        responses = tok.batch_decode(gen, skip_special_tokens=True)
        # compute rewards from static metrics only
        rewards = []
        for resp in responses:
            # try extracting fenced Java, else use full text
            if "```java" in resp:
                code = resp.split("```java")[-1].split("```")[0].strip()
            else:
                code = resp.strip()
            static = metric_bundle(code)
            R = compute_reward(static, weights)
            rewards.append(R)

        # PPO update
        trainer.step(batch["prompt"], responses,
                     torch.tensor(rewards, dtype=torch.float32, device=model.device))

        step_idx = start // ppo_cfg.batch_size
        if step_idx % max(1, args.save_every // max(1, ppo_cfg.batch_size)) == 0:
            trainer.save_pretrained(os.path.join(args.out_dir, f"step_{start}"))
        if step_idx % max(1, args.eval_every // max(1, ppo_cfg.batch_size)) == 0:
            print(f"[eval] step={start} avgR={sum(rewards)/len(rewards):.3f}")

    trainer.save_pretrained(os.path.join(args.out_dir, "final"))
    print("Training complete:", args.out_dir)

if __name__ == "__main__":
    main()
