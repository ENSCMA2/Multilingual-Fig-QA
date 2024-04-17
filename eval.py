import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import *

ALL_LANGS = ["en_dev", "hi", "id", "jv", "kn", "su", "sw"]

for td in ["langdata", "translate-test"]:
	for tf in ALL_LANGS:
		for model in ["llama", "qwen2", "mixtral"]:
			fname = f"preds_{td}_{tf}_{model}.npy"
			preds = np.load(fname)
			trues = pd.read_csv(f"{td}/{tf}.csv")["label"]
			acc = accuracy_score(trues, preds)
			prfs = precision_recall_fscore_support(trues, preds, labels = [0, 1])
			print(td, tf, model, acc, prfs)