#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning a 🤗 Transformers model on multiple choice relying on the accelerate library without using a Trainer.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import argparse
import json
import logging
import math
import os
import copy
import random
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import Optional, Union

import datasets
import torch
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import PaddingStrategy, get_full_repo_name
import pdb

logger = get_logger(__name__)
# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
ALL_LANGS = ["en", "hi", "id", "jv", "kn", "su", "sw"]

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--source_lang",
        type=str,
        help="Source language (to train on)",
        choices=ALL_LANGS
    )
    parser.add_argument(
        "--target_lang",
        type=str,
        help="Target language (to evaluate on)",
        choices=ALL_LANGS
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )

    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )

    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run prediction on the test set. (Training will not be executed.)")

    parser.add_argument("--save_embeddings", action='store_true',
                        help="Whether to save [CLS] embeddings for each example in the test set.")
    
    parser.add_argument("--save_embeddings_in_tsv", action='store_true',
                        help="Whether to save embeddings in tsv format as well.")

    parser.add_argument("--embedding_output_dir", type=str, default=None, help="Where to store embeddings.")

    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activate debug mode and run training only with a subset of data.",
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Disable logging"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        help="Patience for early stopping. If not passed, early stopping is not used."
    )
    args = parser.parse_args()

    # Sanity checks: check whether train and eval file present
    if not args.do_predict:
        if (args.train_file is None or args.validation_file is None) and (args.source_lang is None):
            raise ValueError(
                "Either predict mode or need training and eval file."
            )

    # If predict: check whether test file present
    else:
        if args.test_file is None:
            raise ValueError("Need test file for predict mode.")
    
    # If save embeddings: check whether embedding file present
    if args.save_embeddings:
        if args.embedding_output_dir is None:
            raise ValueError("Need embedding output dir for save embeddings mode.")

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch

def prepare_data():
    pass

def train_model(train_dataloader, model, accelerator, optimizer, lr_scheduler, args, completed_steps, checkpointing_steps, progress_bar, eval_dataloader=None):
    model.train()
    if args["with_tracking"]:
        total_loss = 0
    for step, batch in enumerate(train_dataloader):
        # We need to skip steps until we reach the resumed step
        if args["resume_from_checkpoint"] and epoch == starting_epoch:
            if resume_step is not None and step < resume_step:
                completed_steps += 1
                continue
        outputs = model(input_ids=batch["input_ids"], 
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
        loss = outputs.loss
        # We keep track of the loss at each epoch
        if args["with_tracking"]:
            total_loss += loss.detach().float()
        loss = loss / args["gradient_accumulation_steps"]
        accelerator.backward(loss)
        
        if step % args["gradient_accumulation_steps"] == 0 or step == len(train_dataloader) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if not args["silent"]:
                progress_bar.update(1)
            completed_steps += 1

        if isinstance(checkpointing_steps, int):
            if completed_steps % checkpointing_steps == 0:
                output_dir = f"step_{completed_steps }"
                if args["output_dir"] is not None:
                    output_dir = os.path.join(args["output_dir"], output_dir)
                accelerator.save_state(output_dir)

        if completed_steps >= args["max_train_steps"]:
            break

    return model, loss, completed_steps

def eval_model(model, eval_dataloader, metric, accelerator, epoch, args):
    model.eval()
    samples_seen = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        predictions, references = accelerator.gather((predictions, batch["labels"]))
        # If we are in a multiprocess environment, the last batch has duplicates
        if accelerator.num_processes > 1:
            if step == len(eval_dataloader) - 1:
                predictions = predictions[: len(eval_dataloader.dataset) - samples_seen]
                references = references[: len(eval_dataloader.dataset) - samples_seen]
            else:
                samples_seen += references.shape[0]
        metric.add_batch(
            predictions=predictions,
            references=references,
        )

    eval_metric = metric.compute()
    accelerator.print(f"epoch {epoch}: {eval_metric}")

    return eval_metric

def main_train_loop(train_dataloader, eval_dataloader, model, tokenizer, metric, accelerator, optimizer, lr_scheduler, num_train_epochs, args, starting_epoch=0, checkpointing_steps=None, progress_bar=None):
    completed_steps = 0

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        model, total_loss, completed_steps = train_model(train_dataloader, model, accelerator, optimizer, lr_scheduler, args, completed_steps, checkpointing_steps, progress_bar)

        eval_metric = eval_model(model, eval_dataloader, metric, accelerator, epoch, args)
    
        if args["with_tracking"]:
            accelerator.log(
                {
                    "accuracy": eval_metric,
                    "train_loss": total_loss.item() / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if args["push_to_hub"] and epoch < args["num_train_epochs"] - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                args["output_dir"], is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args["output_dir"])
                repo.push_to_hub(
                    commit_message=f"Training in progress epoch {epoch}", blocking=False, auto_lfs_prune=True
                )

        if args["checkpointing_steps"] == "epoch":
            output_dir = f"epoch_{epoch}"
            if args["output_dir"] is not None:
                output_dir = os.path.join(args["output_dir"], output_dir)
            accelerator.save_state(output_dir)

    if args["output_dir"] is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            args["output_dir"], is_main_process=accelerator.is_main_process, save_function=accelerator.save
        )
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args["output_dir"])
            if args["push_to_hub"]:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)
        with open(os.path.join(args["output_dir"], "all_results.json"), "w") as f:
            json.dump({"eval_accuracy": eval_metric["accuracy"]}, f)

    return eval_metric["accuracy"]

def main(args=None):
    if args is None:
        args = parse_args()
        
    if not isinstance(args, dict):
        args = vars(args)
    
    # or else there will be inconsistencies between hyperparam runs.
    args = copy.deepcopy(args)
    
    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator = (
        Accelerator(log_with=args["report_to"], logging_dir=args["output_dir"]) if args["with_tracking"] else Accelerator()
    )
    # Make one log on every process with the configuration for debugging.
    level = logging.INFO if not args["silent"] else logging.ERROR
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=level,
    )
    logger.info(accelerator.state, main_process_only=False)

    if args["silent"]: # also disable hf logs
        transformers.logging.set_verbosity_error()
        datasets.logging.set_verbosity_error()
        datasets.utils.logging.disable_propagation()
        datasets.disable_progress_bar()
    else:
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args["seed"] is not None:
        set_seed(args["seed"])

    # Handle the repository creation
    if accelerator.is_main_process:
        if args["push_to_hub"]:
            if args["hub_model_id"] is None:
                repo_name = get_full_repo_name(Path(args["output_dir"]).name, token=["hub_token"])
            else:
                repo_name = args["hub_model_id"]
            repo = Repository(args["output_dir"], clone_from=repo_name)

            with open(os.path.join(args["output_dir"], ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args["output_dir"] is not None:
            os.makedirs(args["output_dir"], exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    if args["source_lang"] is not None and args["target_lang"] is not None:
        split_key = "train"
        extension = "csv"
    elif args["do_predict"]:
        split_key = "test"
        extension = args["test_file"].split(".")[-1]
    else:
        split_key = "train"
        extension = args["train_file"].split(".")[-1]

    if args["dataset_name"] is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args["dataset_name"], args["dataset_config_name"])
    else:
        data_files = {}
        if args["source_lang"] is not None and args["target_lang"] is not None:
            data_files["train"] = f'./langdata/{args["source_lang"]}.csv'
            if args["source_lang"] != args["target_lang"]:
                data_files["test"] = f'./langdata/{args["target_lang"]}.csv'
        else:
            if args["train_file"] is not None:
                data_files["train"] = args["train_file"]
            if args["validation_file"] is not None:
                data_files["validation"] = args["validation_file"]
            if args["test_file"] is not None:
                data_files["test"] = args["test_file"]
        raw_datasets = load_dataset(extension, data_files=data_files)

        # split into train and validation if we specified by language
        if args["source_lang"] is not None and args["target_lang"] is not None:
            if args["source_lang"] != args["target_lang"]:
                split_dataset = raw_datasets["train"].train_test_split(test_size=0.1)
                raw_datasets["train"] = split_dataset["train"]
                raw_datasets["validation"] = split_dataset["test"]
            else:
                split_dataset = raw_datasets["train"].train_test_split(test_size=0.1)
                split_dataset_2 = split_dataset["test"].train_test_split(test_size=0.5)
                raw_datasets["train"] = split_dataset["train"]
                raw_datasets["validation"] = split_dataset_2["train"]
                raw_datasets["test"] = split_dataset_2["test"]
    # Trim a number of training examples
    if args["debug"]:
        for split in raw_datasets.keys():
            raw_datasets[split] = raw_datasets[split].select(range(100))
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    column_names = raw_datasets[split_key].column_names

    # When using your own dataset or a different dataset from swag, you will probably need to change this.
    ending_names = [f"ending{i}" for i in [1, 2]]
    context_name = "startphrase"
    label_column_name = "label" if "label" in column_names else "labels"

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if args["config_name"]:
        config = AutoConfig.from_pretrained(args["model_name_or_path"])
    elif args["model_name_or_path"]:
        config = AutoConfig.from_pretrained(args["model_name_or_path"])
    else:
        config = CONFIG_MAPPING[args["model_type"]]()
        logger.warning("You are instantiating a new config instance from scratch.")

    # Set output hidden states to True to get access to all hidden states
    config.output_hidden_states = True

    if args["tokenizer_name"]:
        tokenizer = AutoTokenizer.from_pretrained(args["tokenizer_name"], use_fast=not args["use_slow_tokenizer"])
    elif args["model_name_or_path"]:
        tokenizer = AutoTokenizer.from_pretrained(args["model_name_or_path"], use_fast=not args["use_slow_tokenizer"])
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args["model_name_or_path"]:
        model = AutoModelForMultipleChoice.from_pretrained(
            args["model_name_or_path"],
            from_tf=bool(".ckpt" in args["model_name_or_path"]),
            config=config,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForMultipleChoice.from_config(config)

    model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    padding = "max_length" if args["pad_to_max_length"] else False

    def preprocess_function(examples):
        first_sentences = [[context] * 2 for context in examples[context_name]]
        second_sentences = [[examples[end][i] for end in ending_names] for i in range(len(examples[context_name]))]
        labels = examples[label_column_name]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = tokenizer(
            first_sentences,
            second_sentences,
            max_length=args["max_length"],
            padding=padding,
            truncation=True,
        )

        # Save the decoded sentences if storing embeddings
        if args["do_predict"] and args["save_embeddings"]:
            sentence_fp = os.path.join(args["embedding_output_dir"], "sentences.tsv")
            with open(sentence_fp, "a") as f:
                for i in range(len(tokenized_examples["input_ids"])):
                    f.write(tokenizer.decode(tokenized_examples["input_ids"][i]) + "\n")

        # Un-flatten
        tokenized_inputs = {k: [v[i : i + 2] for i in range(0, len(v), 2)] for k, v in tokenized_examples.items()}
        tokenized_inputs["labels"] = labels

        return tokenized_inputs

    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function, batched=True, remove_columns=raw_datasets[split_key].column_names
        )

    # DataLoaders creation:
    if args["pad_to_max_length"]:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorForMultipleChoice(
            tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None)
        )
    
    # Metrics
    metric = load_metric("accuracy")

    if not args["do_predict"]:
        train_dataset = processed_datasets["train"]
        eval_dataset = processed_datasets["validation"]

        # Log a few random samples from the training set:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args["per_device_train_batch_size"]
        )
        eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args["per_device_eval_batch_size"])

        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args["weight_decay"],
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args["learning_rate"])

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args["gradient_accumulation_steps"])

        # This section of code is causing a strange error when re-run.
        if args["max_train_steps"] is None:
            args["max_train_steps"] = args["num_train_epochs"] * num_update_steps_per_epoch
        else:
            args["num_train_epochs"] = math.ceil(args["max_train_steps"] / num_update_steps_per_epoch)

        num_warmup_steps = 0.1 * args["max_train_steps"]
        lr_scheduler = get_scheduler(
            name=args["lr_scheduler_type"],
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=args["max_train_steps"],
        )

        # Prepare everything with our `accelerator`.
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
        )

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args["gradient_accumulation_steps"])
        args["max_train_steps"] = args["num_train_epochs"] * num_update_steps_per_epoch

        # Figure out how many steps we should save the Accelerator states
        if hasattr(args["checkpointing_steps"], "isdigit"):
            checkpointing_steps = args["checkpointing_steps"]
            if args["checkpointing_steps"].isdigit():
                checkpointing_steps = int(args["checkpointing_steps"])
        else:
            checkpointing_steps = None

        # We need to initialize the trackers we use, and also store our configuration.
        # We initialize the trackers only on main process because `accelerator.log`
        # only logs on main process and we don't want empty logs/runs on other processes.
        if args["with_tracking"]:
            if accelerator.is_main_process:
                if not isinstance(args, dict):
                    experiment_config = vars(args)
                # TensorBoard cannot log Enums, need the raw value
                experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
                accelerator.init_trackers("swag_no_trainer", experiment_config)

        # Train!
        total_batch_size = args["per_device_train_batch_size"] * accelerator.num_processes * args["gradient_accumulation_steps"]
        
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args['num_train_epochs']}")
        logger.info(f"  Instantaneous batch size per device = {args['per_device_train_batch_size']}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args['gradient_accumulation_steps']}")
        logger.info(f"  Total optimization steps = {args['max_train_steps']}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(args["max_train_steps"]), disable=not accelerator.is_local_main_process)
        #completed_steps = 0
        starting_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args["resume_from_checkpoint"]:
            if args["resume_from_checkpoint"] is not None or args["resume_from_checkpoint"] != "":
                accelerator.print(f"Resumed from checkpoint: {args['resume_from_checkpoint']}")
                accelerator.load_state(args["resume_from_checkpoint"])
                path = os.path.basename(args["resume_from_checkpoint"])
            else:
                # Get the most recent checkpoint
                dirs = [f.name for f in os.scandir(os.getcwd()) if f.is_dir()]
                dirs.sort(key=os.path.getctime)
                path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last
            # Extract `epoch_{i}` or `step_{i}`
            training_difference = os.path.splitext(path)[0]

            if "epoch" in training_difference:
                starting_epoch = int(training_difference.replace("epoch_", "")) + 1
                resume_step = None
            else:
                resume_step = int(training_difference.replace("step_", ""))
                starting_epoch = resume_step // len(train_dataloader)
                resume_step -= starting_epoch * len(train_dataloader)

        num_train_epochs = args["num_train_epochs"] if not args["early_stopping_patience"] else 1000
        acc_final = main_train_loop(train_dataloader, eval_dataloader, model, tokenizer, metric, accelerator, optimizer, lr_scheduler, args["num_train_epochs"], args, starting_epoch=0, checkpointing_steps=checkpointing_steps)
        return acc_final
    else:
         # Use the device given by the `accelerator` object.
        cls_embeddings_all = []
        test_dataset = processed_datasets["test"]
        test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args["per_device_eval_batch_size"])
        model, test_dataloader = accelerator.prepare(model, test_dataloader)
        for step, batch in enumerate(test_dataloader):
            with torch.no_grad():
                outputs = model(input_ids=batch["input_ids"], 
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"])
            predictions = outputs.logits.argmax(dim=-1)
            predictions, references = accelerator.gather((predictions, batch["labels"]))

            # Obtain [CLS] embedding from hidden states
            if args["save_embeddings"]:
                tensor_outfile = os.path.join(args["embedding_output_dir"], "embeddings.pt")
                tsv_outfile = os.path.join(args["embedding_output_dir"], "embeddings.tsv")
                cls_embeddings = outputs.hidden_states[-1][:, 0, :]
                cls_embeddings = accelerator.gather(cls_embeddings)
                if accelerator.is_main_process:
                    for cls_embedding in cls_embeddings:
                        cls_embeddings_all.append(cls_embedding)
                        cls_embedding = cls_embedding.cpu().numpy()
                        with open(tsv_outfile, "a") as f:
                            f.write("\t".join([str(x) for x in cls_embedding]) + "\n")
            # If we are in a multiprocess environment, the last batch has duplicates
            if accelerator.num_processes > 1:
                if step == len(test_dataloader) - 1:
                    predictions = predictions[: len(test_dataloader.dataset) - samples_seen]
                    references = references[: len(test_dataloader.dataset) - samples_seen]
                else:
                    samples_seen += references.shape[0]
            metric.add_batch(
                predictions=predictions,
                references=references,
            )

        eval_metric = metric.compute()
        accelerator.print(f"test {args['test_file']}: {eval_metric}")

        if args["save_embeddings"]:
            if accelerator.is_main_process:
                torch.save({"embeddings": cls_embeddings_all}, tensor_outfile)


if __name__ == "__main__":
    main()