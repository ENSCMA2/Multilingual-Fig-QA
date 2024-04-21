import requests
from bs4 import BeautifulSoup
import re

suffs = ["1", "1?page=1", "2728", "200"]
def clean(text):
		return text.replace("Research Areas:", "").replace("Office:", "").replace("Phone:", "").strip()

for page in suffs:
	r = requests.get(f"https://www.lti.cs.cmu.edu/directory/all/154/{page}")

	soup = BeautifulSoup(r.text, features="html.parser")
	paget = soup.head.title.text
	entries = soup.find_all("td", {"class", re.compile(r'col-\d col-*')})
	print(len(entries))
	for entry in entries:
		if entry.h2 is None:
			continue
		name = entry.h2.text
		try:
			title = entry.find_all("div", {"class": "views-field views-field-field-computed-prof-title"})[0].div.text
		except:
			title = ""
		try:
			email = entry.find_all("a", {"href": re.compile(r'mailto*')})[0].text
		except:
			email = ""
		try:
			office = clean(entry.find_all("div", {"class": "views-field views-field-field-computed-building"})[0].text)
		except:
			office = ""
		try:
			phone = clean(entry.find_all("div", {"class": "views-field views-field-field-computed-phone"})[0].text)
		except:
			phone = ""
		try:
			interests = clean(entry.find_all("div", {"class": "views-field views-field-field-research-areas"})[0].text)
		except:
			interests = ""
		formatted = f"Name: {name}\nTitle: {title}\nEmail {email}\n Phone: {phone}\nOffice: {office}\nInterests: {interests}"
		with open(f"../../collection/directory/{paget}_{name}.txt", "w") as o:
			o.write(formatted)