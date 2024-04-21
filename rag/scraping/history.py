import requests
import os
from bs4 import BeautifulSoup
from pypdf import PdfReader

urls = ["https://www.cs.cmu.edu/scs25/25things", "https://www.cs.cmu.edu/scs25/history", "https://www.cmu.edu/about/history.html"]

'''
for url in urls:
	things_25 = BeautifulSoup(requests.get(url).text, features="html.parser")
	with open(f"../../collection/history/cmu/{url.split('/')[-1]}.txt", "w") as o:
		o.write(things_25.head.text.strip() + "\n" + things_25.body.text.strip())
'''

reader = PdfReader(f"cmu_fact_sheet_02.pdf")
text = ""
for page in reader.pages:
	text += (page.extract_text())
with open(f"../../collection/history/cmu/factsheet.txt", "w") as o:
	o.write(text)

