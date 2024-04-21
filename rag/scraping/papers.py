import time
import requests
import os

# title, abstract, authors, publication venue, year and tldr

def contains_all_parts(candidate, parts):
	for part in parts:
		if part not in candidate:
			return False
	return True

def contains_some_parts(candidate, parts):
	for part in parts:
		if part in candidate:
			return True
	return False

def process(value, key):
	if value is None:
		return ""
	if key == "tldr":
		return value["text"]
	if key == "publicationVenue":
		return value["name"]
	if key == "authors":
		return "; ".join([auth["name"] for auth in authors])
	return value
all_names = []

for file in os.listdir("../../collection/directory"):
	name = file.split("_")[-1].split(".")[0]
	all_names.append(name)
	name_parts = name.split()
	id_query = f"https://api.semanticscholar.org/graph/v1/author/search?query={name}"
	author_ids = requests.get(id_query).json()["data"]
	if len(author_ids) == 1:
		author_id = author_ids[0]["authorId"]
	else:
		plausible_ids = [i for i in author_ids if contains_all_parts(i["name"], name_parts)]
		if len(plausible_ids) > 0:
			author_id = plausible_ids[0]["authorId"]
		else:
			plausible_ids = [i for i in author_ids if contains_some_parts(i["name"], name_parts)]
			author_id = plausible_ids[0]["authorId"]
	author_query = f"https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"
	papers = requests.get(author_query).json()["data"]
	print(name, author_id, len(papers))
	time.sleep(1)
	for paper_id in papers:
		if not os.path.exists(f"../../collection/papers/{name}_{paper_id['paperId']}.txt"):
			paper_query = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id['paperId']}?fields=title,abstract,authors,publicationVenue,year,tldr,isOpenAccess"
			metadata = requests.get(paper_query).json()
			print(metadata)
			if metadata["isOpenAccess"] and int(metadata["year"]) == 2023:
				formatted = ""
				for key in metadata:
					if key in ["title", "abstract", "authors", "publicationVenue", 
							   "year", "tldr"]:
						formatted += f"{key}: {process(metadata[key], key)}\n"
				with open(f"../../collection/papers/{name}_{metadata['title']}.txt", "w") as o:
					o.write(formatted)
			time.sleep(1)

missing = [('Jamie Callan', 144987107), 
		   ('Norman Sadeh', 2464164), 
		   ('Kemal Oflazer', 1723120),
		   ('Lei Li', 143900005),
		   ('Ravi Starzl', 5000465),
		   ('Thomas Schaaf', 145849024),
		   ('Michael Mauldin', 35497738),
		   ('Rodolfo M Vega', 151196614),
		   ('Daphne Ippolito', 7975935),
		   ('Rita Singh', 153915824),
		   ('Lu Jiang', 39978626),
		   ('Michael Shamos', 1890127),
		   ('Daniel Fried', 47070750),
		   ('Florian Metze', 1740721),
		   ('Raj Reddy', 145502114),
		   ('Anatole Gershman', 145001267),
		   ('Yiming Yang', 35729970),
		   ('Matthias Grabmair', 2869551),
		   ('William Cohen', 50056360),
		   ('Christopher Dyer', 1745899), 
		   ('Teruko Mitamura', 1706595),
		   ('Monika Woszczyna', 2510215),
		   ('Fernando Diaz', 145472333),
		   ('Shinji Watanabe', 1746678),
		   ('Brian MacWhinney', 2414040),
		   ('Robert Frederking', 2260563),
		   ('Ian Lane', 1765892),
		   ('Eric P Xing', 143977260),
		   ('Alexander Waibel', 1724972), 
		   ('Jack Mostow', 1695106),
		   ('Sean Welleck', 2129663),
		   ('Ralf Brown', 2109449533),
		   ('Madhavi Ganapathiraju', 49874057), 
		   ('Alon Lavie', 1784914),
		   ('Lori Levin', 1686960)]

for name, author_id in missing:
	author_query = f"https://api.semanticscholar.org/graph/v1/author/{author_id}/papers"
	papers = requests.get(author_query).json()["data"]
	for paper_id in papers:
		if not os.path.exists(f"../../collection/papers/{name}_{paper_id['paperId']}.txt"):
			paper_query = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id['paperId']}?fields=title,abstract,authors,publicationVenue,year,tldr,isOpenAccess"
			metadata = requests.get(paper_query).json()
			if metadata["isOpenAccess"] and int(metadata["year"]) == 2023:
				formatted = ""
				for key in metadata:
					if key in ["title", "abstract", "authors", "publicationVenue", 
							   "year", "tldr"]:
						formatted += f"{key}: {process(metadata[key], key)}\n"
				with open(f"../../collection/papers/{name}_{paper_id['paperId']}.txt", "w") as o:
					o.write(formatted)
			time.sleep(1)
