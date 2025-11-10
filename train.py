from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch
from trl import GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig
from trl import GRPOTrainer


PROMPT_TMPL = (
    "You are a helpful Java coding assistant.\n"
    "Task: {task}\n"
    "Signature: {sig}\n"
    "Return ONLY Java code.\n"
)


def to_query(ex):
    task = ex.get("human_label") or ex.get("docstring") or ""
    sig = ex.get("name") or ""
    return {"prompt": PROMPT_TMPL.format(task=task.strip(), sig=sig.strip())}


def cyclomatic_like(code: str) -> int:
    return (
        1
        + sum((" if " in code,))
        + code.count(" for(")
        + code.count(" while(")
        + code.count(" case ")
        + code.count(" catch(")
    )


def comment_ratio(code: str) -> float:
    lines = [l for l in code.splitlines() if l.strip()]
    if not lines:
        return 0.0
    c = sum(
        1
        for l in lines
        if l.strip().startswith("//") or "/*" in l or "*/" in l or l.strip() == "*"
    )
    return min(1.0, c / len(lines))


def duplication_ratio(code: str, win=10) -> float:
    toks = code.replace("\n", " ").split()
    if len(toks) < 2 * win:
        return 0.0
    seen = {}
    for i in range(len(toks) - win + 1):
        k = tuple(toks[i : i + win])
        seen[k] = seen.get(k, 0) + 1
    dup = sum((v - 1) * win for v in seen.values() if v > 1)
    return min(1.0, dup / max(1, len(toks)))


def norm(x, lo, hi, invert=False):
    y = (float(x) - lo) / max(1e-9, hi - lo)
    y = max(0.0, min(1.0, y))
    return 1.0 - y if invert else y


def extract_java(text: str) -> str:
    # strip everything up to instruction line
    if "Return ONLY Java code." in text:
        text = text.split("Return ONLY Java code.", 1)[-1].strip()
    if "```java" in text:
        return text.split("```java")[-1].split("```")[0].strip()
    return text.strip()


def reward_fn(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    rewards = []
    for t in completions:
        code = extract_java(t)
        cc = cyclomatic_like(" " + code.replace("\n", " ") + " ")
        loc = len([l for l in code.splitlines() if l.strip()])
        comm = comment_ratio(code)
        dup = duplication_ratio(code)

        # normalize (tune ranges for your corpus if needed)
        m_complex = (norm(cc, 1, 20, True) + norm(loc, 5, 120, True)) / 2.0
        m_mod = norm(dup, 0.0, 0.15, True)
        m_read = norm(comm, 0.00, 0.30, False)

        R = 0.4 * m_complex + 0.3 * m_mod + 0.3 * m_read  # [0..1]
        rewards.append(float(R))

    return rewards


output_dir = "Qwen2.5-1.5B-Instruct-trl-grpo"

# Configure training arguments using GRPOConfig
training_args = GRPOConfig(
    learning_rate=2e-5,
    # num_train_epochs=1,
    max_steps=100,  # Number of dataset passes. For full trainings, use `num_train_epochs` instead
    # Parameters that control the data preprocessing
    per_device_train_batch_size=2,
    max_completion_length=1024,  # default: 256            # Max completion length produced during training
    num_generations=2,  # 2, # default: 8                  # Number of generations produced during trainig for comparison
    max_prompt_length=2048,  # default: 512                # Max prompt lenght of the input prompt used for generation during training
    fp16=True,
    # Parameters related to reporting and saving
    output_dir=output_dir,  # Where to save model checkpoints and logs
    logging_steps=1,  # Log training metrics every N steps
    report_to="tensorboard",  # Experiment tracking tool
)


def main():
    data_path = "./data/CoderEval4Java.json"
    model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"

    # Загружаем JSON
    dataset = load_dataset("json", data_files=data_path, field="RECORDS")

    # Проверим структуру
    print(dataset)
    print(dataset["train"][0])

    dataset = dataset.map(to_query, remove_columns=dataset["train"].column_names)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto"
    )

    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"],
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_fn],
        args=training_args,
        train_dataset=dataset["train"],
        peft_config=peft_config,
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)

    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)

    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime'] / 60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")


if __name__ == "__main__":
    main()
