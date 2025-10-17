from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Dict, List, Sequence

from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune GPT-2 on WikiText-103 with the Hugging Face Trainer."
    )
    parser.add_argument(
        "--model-name",
        default="gpt2",
        help="Base model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--dataset-name",
        default="wikitext",
        help="Dataset repository on the Hugging Face Hub.",
    )
    parser.add_argument(
        "--dataset-config",
        default="wikitext-103-raw-v1",
        help="Configuration name for the dataset.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=1024,
        help="Maximum sequence length after tokenization.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/gpt2-wikitext"),
        help="Where checkpoints and logs should be written.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=1.0,
        help="Total number of training epochs to run.",
    )
    parser.add_argument(
        "--train-batch-size",
        type=int,
        default=2,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2,
        help="Per-device evaluation batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Accumulate gradients over this many steps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer.",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        help="Number of steps used for a linear warmup.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=50,
        help="Log training metrics every X steps.",
    )
    parser.add_argument(
        "--save-total-limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep on disk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed that is passed to the Trainer.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        default="linear",
        help="Learning rate scheduler type. See transformers.SchedulerType for options.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable mixed precision training in float16.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Enable mixed precision training in bfloat16.",
    )
    parser.add_argument(
        "--report-to",
        default="",
        help="Comma separated list of reporting integrations (e.g. tensorboard). Leave empty to disable.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Path to a checkpoint directory to resume from.",
    )
    parser.add_argument(
        "--overwrite-output-dir",
        action="store_true",
        help="Overwrite the contents of the output directory if it already exists.",
    )
    parser.add_argument(
        "--use-slow-tokenizer",
        action="store_true",
        help="Load the slow Python tokenizer instead of the Rust implementation.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )


def tokenize_dataset(
    raw_datasets: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    block_size: int,
) -> DatasetDict:
    def tokenize_function(examples: Dict[str, Sequence[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(examples["text"])

    tokenized = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )

    def group_texts(examples: Dict[str, List[List[int]]]) -> Dict[str, List[List[int]]]:
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated["input_ids"])
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return tokenized.map(
        group_texts,
        batched=True,
        desc=f"Grouping tokens into blocks of {block_size}",
    )


def safe_exp(value: float) -> float:
    try:
        return math.exp(value)
    except OverflowError:
        return float("inf")


def main() -> None:
    args = parse_args()
    configure_logging()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    last_checkpoint = None
    if args.output_dir.exists() and not args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(str(args.output_dir))
        if last_checkpoint is not None:
            LOGGER.info("Resuming from last checkpoint at %s", last_checkpoint)

    LOGGER.info(
        "Loading dataset %s (%s)",
        args.dataset_name,
        args.dataset_config,
    )
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=not args.use_slow_tokenizer,
    )

    if tokenizer.pad_token is None:
        LOGGER.info(
            "Tokenizer has no pad token. Using eos_token (%s) as pad.",
            tokenizer.eos_token,
        )
        tokenizer.pad_token = tokenizer.eos_token

    tokenized_datasets = tokenize_dataset(raw_datasets, tokenizer, args.block_size)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model.resize_token_embeddings(len(tokenizer))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    report_to = [item.strip() for item in args.report_to.split(",") if item.strip()]

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=args.overwrite_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=args.fp16,
        bf16=args.bf16,
        seed=args.seed,
        report_to=report_to,
        push_to_hub=False,
    )

    LOGGER.info("Starting training with %s training sequences", len(train_dataset))
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    checkpoint = args.resume_from_checkpoint or last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()

    train_metrics = train_result.metrics
    if "train_loss" in train_metrics:
        train_metrics["train_perplexity"] = safe_exp(train_metrics["train_loss"])
    trainer.log_metrics("train", train_metrics)
    trainer.save_metrics("train", train_metrics)
    trainer.save_state()

    eval_metrics = trainer.evaluate()
    if "eval_loss" in eval_metrics:
        eval_metrics["perplexity"] = safe_exp(eval_metrics["eval_loss"])
    trainer.log_metrics("eval", eval_metrics)
    trainer.save_metrics("eval", eval_metrics)

    LOGGER.info("Training complete. Metrics written to %s", args.output_dir)


if __name__ == "__main__":
    main()
