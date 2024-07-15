# Editor:CharPy
# Edit time:2024/5/16 19:15

import csv
dataGraph = []
with open('relations.txt', 'r', encoding = 'utf-8') as f:
	lines = f.readlines()
	for item in lines:
		item = item[:-1].split(' ----- ')
		dataGraph.append(item)
		print(item)
with open('relations.csv', 'w', encoding = 'utf-8', newline = '') as f:
	writer = csv.writer(f)
	writer.writerow(["entity1", "entity2", "relation"])
	for item in dataGraph:
		writer.writerow(item)