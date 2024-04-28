import pandas as pd
import os
import sys

retriever = sys.argv[1]

root = f"../experiment/{retriever}"

eventual_df = {"langdata": [], "translate-test": []}
metadata = {"lang": [], "model": []}
for lang in ["jv", "kn", "su", "sw"]:
	for model in ["meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.2", "Qwen/Qwen1.5-7B-Chat"]:
		mstr = model.replace("/", "-")
		metadata["lang"].append(lang)
		metadata["model"].append(model)
		for t in ["langdata", "translate-test"]:
			pth = f"{root}/{lang}/{mstr}/{t}/stats.csv"
			result = pd.read_csv(pth)["Accuracy"].tolist()[0]
			eventual_df[t].append(result)			

df = pd.DataFrame({"Language": metadata["lang"], "Model": metadata["model"],
				   "Accuracy": eventual_df["langdata"], 
				   "Translated Accuracy": eventual_df["translate-test"]})

df.to_csv(f"{retriever}_summary.csv")
