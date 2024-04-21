import pandas as pd
import os
import json

'''
publicationVenue: {'id': '1901e811-ee72-4b20-8f7e-de08cd395a10', 'name': 'arXiv.org', 'alternate_names': ['ArXiv'], 'issn': '2331-8422', 'url': 'https://arxiv.org'}
title: Multilingual Machine Translation with Large Language Models: Empirical Results and Analysis
abstract: Large language models (LLMs) have demonstrated remarkable potential in handling multilingual machine translation (MMT). In this paper, we systematically investigate the advantages and challenges of LLMs for MMT by answering two questions: 1) How well do LLMs perform in translating massive languages? 2) Which factors affect LLMs' performance in translation? We thoroughly evaluate eight popular LLMs, including ChatGPT and GPT-4. Our empirical results show that translation capabilities of LLMs are continually improving. GPT-4 has beat the strong supervised baseline NLLB in 40.91% of translation directions but still faces a large gap towards the commercial translation system, especially on low-resource languages. Through further analysis, we discover that LLMs exhibit new working patterns when used for MMT. First, instruction semantics can surprisingly be ignored when given in-context exemplars. Second, cross-lingual exemplars can provide better task guidance for low-resource translation than exemplars in the same language pairs. Third, LLM can acquire translation ability in a resource-efficient way and generate moderate translation even on zero-resource languages.
year: 2023
tldr: It is discovered that LLMs exhibit new working patterns when used for MMT and cross-lingual exemplars can provide better task guidance for low-resource translation than exemplars in the same language pairs.
authors: [{'authorId': '2131383723', 'name': 'Wenhao Zhu'}, {'authorId': '2115669628', 'name': 'Hongyi Liu'}, {'authorId': '2047143813', 'name': 'Qingxiu Dong'}, {'authorId': '47883405', 'name': 'Jingjing Xu'}, {'authorId': '47648549', 'name': 'Lingpeng Kong'}, {'authorId': '1838162', 'name': 'Jiajun Chen'}, {'authorId': '143900005', 'name': 'Lei Li'}, {'authorId': '2046010', 'name': 'Shujian Huang'}]
'''

megainfo = {}
venues = []
authors = []
questions = []
answers = []

def jload(stuff):
	repstuff = stuff.replace("'", '"').replace('s"e', "se").replace('G"S', "GS").replace('l"e', "le").replace('i"c', "ic").replace('c"i', "ci").replace('R"i', "Ri")
	repstuff = repstuff.replace("None", '"0"').replace('r"i', "ri")
	return json.loads(repstuff)

def name(auth):
	if type(auth) == dict:
		return auth["name"]
	return auth

for file in os.listdir("../../collection/papers"):
	lines = []
	with open(f"../../collection/papers/{file}") as o:
		for line in o:
			lines.append(str(line).strip())
	info = {}
	for line in lines:
		split_by_colon = line.split(":")
		key = split_by_colon[0].strip()
		value = ":".join(split_by_colon[1:]).strip()
		if len(value) > 0:
			info[key] = value
	megainfo[info["title"]] = info

for title in megainfo.keys():
	info = megainfo[title]
	authors = jload(info["authors"])
	questions.append(f"Who were the authors of the paper called {title}?")
	answers.append(", ".join([name(author) for author in authors]))
	questions.append(f"Who was the first author of the paper called {title}?")
	answers.append(authors[0]["name"])
	questions.append(f"Who was the last author of the paper called {title}?")
	answers.append(authors[-1]["name"])
	if "publicationVenue" in info.keys():
		answers.append(name(jload(info["publicationVenue"])))
		questions.append(f"At what venue was the paper called {title} published?")
	if "tldr" in info.keys():
		questions.append(f"What was the paper called {title} about?")
		answers.append(info["tldr"])
	if len(info["year"].strip()) > 0:
		questions.append(f"What year was the paper called {title} published?")
		answers.append(info["year"])

'''
for author in set(authors):
	questions.append(f"What papers did {author} publish in 2023?")
	answer = []
	for title in megainfo.keys():
		info = megainfo[name]
		authors_sub = [auth["name"] for auth in jload(info["authors"])]
		if author in authors_sub:
			answer.append(title)
	answers.append(", ".join(answer))
	# for venue in set(venues):
		# questions.append(f"What papers did {author} publish in the venue {venue}?")
		# answers.append(", ".join([title for title in answer if jload(megainfo[title]["publicationVenue"])["name"] == venue]))
'''
pd.DataFrame({"Q": questions, 
			  "A": answers}).to_csv("../../annotations/papersQAs.tsv", 
			   						sep = "\t")
