import pandas as pd
import os

'''
Name: Alon Lavie
Title: Vice President of Language Technologies at Unbabel and Consulting Professor at the Language Technologies Institute
Email alavie@cs.cmu.edu
 Phone: 
Office: 
Interests: Machine Translation, Natural Language Processing and Computational Linguistics
'''

def ask_about_email(name):
	return f"What is {name}'s email?"

def ask_about_phone(name):
	return f"What is {name}'s phone number?"

def ask_about_office(name):
	return f"Where is {name}'s office?"

def ask_about_title(name):
	return f"What is {name}'s professional title?"

def ask_about_interests(name):
	return f"What are {name}'s research interests?"

def ask_who_interest(interest):
	return f"Which faculty are interested in {interest}?"

def ask_who_title(title):
	return f"Which faculty have the job title {title}?"

megainfo = {}
interests = []
titles = []
questions = []
answers = []
for file in os.listdir("../../collection/directory"):
	lines = []
	with open(f"../../collection/directory/{file}") as o:
		for line in o:
			lines.append(str(line).strip())
	info = {}
	for line in lines:
		if ":" in line:
			split_by_colon = line.split(":")
			key = split_by_colon[0].lower().strip()
			value = split_by_colon[-1].strip()
			if len(value) > 0:
				info[key] = value
		else:
			split_by_space = line.split()
			key = split_by_space[0].lower().strip()
			value = split_by_space[-1].strip()
			if len(value) > 0:
				info[key] = value
		if "interests" in info.keys():
			ints = info["interests"].split(", ")
			interests.extend(ints)
		if "title" in info.keys():
			titles.append(info["title"])
	megainfo[info["name"]] = info

for name in megainfo.keys():
	info = megainfo[name]
	if "email" in info.keys():
		questions.append(ask_about_email(name))
		answers.append(info["email"])
	if "phone" in info.keys():
		questions.append(ask_about_phone(name))
		answers.append(info["phone"])
	if "office" in info.keys():
		questions.append(ask_about_office(name))
		answers.append(info["office"])
	if "title" in info.keys():
		questions.append(ask_about_title(name))
		answers.append(info["title"])
	if "interests" in info.keys():
		questions.append(ask_about_interests(name))
		answers.append(info["interests"])

for interest in set(interests):
	questions.append(ask_who_interest(interest))
	answer = []
	for name in megainfo.keys():
		info = megainfo[name]
		if "interests" in info.keys() and interest in info["interests"]:
			answer.append(name)
	answers.append(", ".join(answer))

for title in set(titles):
	questions.append(ask_who_title(title))
	answer = []
	for name in megainfo.keys():
		info = megainfo[name]
		if "title" in info.keys() and info["title"] == title:
			answer.append(name)
	answers.append(", ".join(answer))

pd.DataFrame({"Q": questions, 
			  "A": answers}).to_csv("../../annotations/directoryQAs.tsv", 
			   						sep = "\t")
