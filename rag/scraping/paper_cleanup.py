import pandas as pd
import json
import os

def jload(stuff):
	repstuff = stuff.replace("'", '"').replace('s"e', "se").replace('G"S', "GS").replace('l"e', "le").replace('i"c', "ic").replace('c"i', "ci").replace('R"i', "Ri")
	repstuff = repstuff.replace("None", '"0"').replace('r"i', "ri")
	return json.loads(repstuff)

def name(auth):
	if type(auth) == dict:
		return auth["name"]
	return auth

def process(value, key):
	if value is None:
		return ""
	if key == "publicationVenue":
		return name(jload(value))
	if key == "authors":
		return ", ".join([name(auth) for auth in jload(value)])
	return value

for file in os.listdir("../../collection/papers_old"):
	lines = []
	with open(f"../../collection/papers_old/{file}") as o:
		for line in o:
			lines.append(str(line).strip())
	cmuauthor = file.split("_")[0]
	info = {}
	for line in lines:
		split_by_colon = line.split(":")
		key = split_by_colon[0].strip()
		value = ":".join(split_by_colon[1:]).strip()
		if len(value) > 0:
			info[key] = process(value, key)
	ptitle = info["title"]
	with open(f"../../collection/papers/{cmuauthor}_{ptitle}.txt", "w") as o:
		for key in info:
			value = info[key]
			o.write(f"{key}: {value}\n")