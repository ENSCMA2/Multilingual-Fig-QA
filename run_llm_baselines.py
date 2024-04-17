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
import numpy as np
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
import pandas as pd
import datasets
import torch
from datasets import load_dataset
import evaluate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import time
import transformers
from accelerate import *
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
from openai import OpenAI
import os
from transformers import LlamaForCausalLM, LlamaTokenizer, Qwen2ForCausalLM, AutoModelForCausalLM
from transformers.utils import PaddingStrategy, get_full_repo_name
import pdb
import re
import replicate
import requests
TOGETHER_API_KEY = os.environ.get("TOGETHER_KEY")
client = OpenAI(
  api_key=TOGETHER_API_KEY,
  base_url='https://api.together.xyz/v1',
)
def model_short(mname):
    if "llama" in mname:
        return "llama"
    if "mistral" in mname:
        return "mistral"
    return "qwen2"

ALL_LANGS = ["en", "hi", "id", "jv", "kn", "su", "sw", None]

def parse_args():
    parser = argparse.ArgumentParser(description="Prompt an LLM on a text classification task")
    parser.add_argument(
        "--test_file", type=str, default=None, help="Name without extension of file containing eval data."
    )
    parser.add_argument(
        "--test_dir", type=str, default=None, help="Directory of file containing eval data."
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=False,
    )
    parser.add_argument("--n", type=int, help="Number of ICL examples", required=False)
    args = parser.parse_args()
    return args

def prompt_template(batch, lang, n = 0):
    q = batch["startphrase"]
    o1 = batch["ending1"]
    o2 = batch["ending2"]
    if n == 0:
        shots = ""
    else:
        td = f"data/addition_data_winogrand/train/{lang}/{lang}_{n}.csv"
        relevant = pd.read_csv(td).iloc[-2 * n:, :]
        strings = []
        for i, tem in relevant.iterrows():
            string = f'''\nPhrase: {tem['startphrase']}
ending1: {tem['ending1']}
ending2: {tem['ending2']}
{'ending1' if tem['labels'] == 0 else 'ending2'}'''
            strings.append(string)
        shots = f'''\nHere are some examples of phrases, possible endings, and which ending corresponds to the meaning of each phrase.{''.join(strings)}
Now, it's your turn.'''
    sys = f"You are completing a multiple-choice task. Answer with a single integer."
    u = f''''{q}'{shots}
Which of the following corresponds to the meaning of the phrase above, enclosed in single quotes?
1: {o1}
2: {o2}
Respond with a single number: 1 or 2.'''
    return sys, u

def eval_model(model, dataset, name, mname, n = 0, lang = None):
    correct = 0
    total = 0
    preds = []
    if os.path.exists(f"preds_{name}_{model_short(mname)}"):
        preds = np.load(f"preds_{name}_{model_short(mname)}").tolist()
    for i, batch in dataset.iterrows():
        if i >= len(preds):
            sys, u = prompt_template(batch, lang, n)
            predictions = client.chat.completions.create(
                              messages=[
                                {
                                  "role": "system",
                                  "content": sys,
                                },
                                {
                                  "role": "user",
                                  "content": u,
                                }
                              ],
                              model=model,
                              max_tokens = 20,
                            ).choices[0].message.content
            print(i, dataset.shape[0])
            print("SYS", sys)
            print("U", u)
            print("PRED", predictions)
            references = batch["labels"]
            if "1" in predictions.lower():
                pred = 0
            else:
                pred = 1
            total += 1
            correct += pred == references
            preds.append(pred)
            np.save(f"preds_{name}_{model_short(mname)}", preds)
            time.sleep(0.5)
    np.save(f"preds_{name}_{model_short(mname)}", preds)

    return correct / total

def main(args=None):
    if args is None:
        args = parse_args()    
    if not isinstance(args, dict):
        args = vars(args)
    # or else there will be inconsistencies between hyperparam runs.
    args = copy.deepcopy(args)
    lang = args["test_file"].split("_")[0]
    
    dataset = pd.read_csv(f"{args['test_dir']}/{args['test_file']}.csv")
    rf = f"preds_{args['test_dir']}_{args['test_file']}_{model_short(args['model_name_or_path'])}.npy"
    if os.path.exists(rf) and len(np.load(rf)) >= dataset.shape[0]:
        print("already exists")
        return

    acc = eval_model(args["model_name_or_path"], dataset, f"{args['test_dir']}_{args['test_file']}", args["model_name_or_path"], n = args['n'], lang = lang)
    print(f"{args['model_name_or_path']}, {args['test_dir']}_{args['test_file']}: {acc}")

if __name__ == "__main__":
    main()
