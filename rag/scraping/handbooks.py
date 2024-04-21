from pypdf import PdfReader


start_end = {"PhD_Student_Handbook_2023-2024": 
			 [(8, 12), (11, 14), (13, 23), (22, 32), (31, 33), (32, 39), 
			  (39, 49)],
			 "MLT Student Handbook 2023 - 2024":
			 [(5, 10), (9, 11), (10, 12), (12, 20), (19, 27), (26, 29), 
			  (28, 36), (35, 48)],
			 "MIIS Handbook_2023 - 2024":
			 [(5, 10), (9, 11), (10, 12), (12, 20), (19, 23), (22, 27), 
			  (26, 30), (29, 32), (31, 38), (38, 48)],
			 "MCDS Handbook 23-24 AY":
			 [(6, 11), (10, 12), (11, 27), (26, 33), (32, 36), (35, 44), 
			  (44, 56)],
			 "Handbook-MSAII-2022-2023":
			 [(5, 6), (5, 6), (5, 7), (6, 8), (7, 8), (7, 8), (7, 9), (8, 11),
			  (10, 11), (11, 23), (22, 33), (32, 34), (34, 41), (40, 53)]}
for key in start_end:
	reader = PdfReader(f"{key}.pdf")
	for i in range(len(start_end[key])):
		start_page, end_page = start_end[key][i]
		pages = reader.pages[start_page:end_page]
		text = "\n".join([page.extract_text() for page in pages])
		with open(f"../../collection/programs/handbooks/{key}_{i}.txt", "w") as o:
			o.write(f"{key}\n")
			o.write(text)