# Editor:CharPy
# Edit time:2024/5/16 16:13
import csv
import py2neo
from py2neo import Graph, Node, Relationship, NodeMatcher

# MATCH(n:Entity) Detach Delete n

# 账号密码改为自己的即可
g = Graph('bolt://localhost:7687', user = 'neo4j', password = 'Pine20210238')
with open('relations.csv', 'r', encoding = 'utf-8') as f:
	reader=csv.reader(f)
	for item in reader:
		if reader.line_num == 1:
			continue
		print( "当前内容：", item)
		start_node = Node("Entity", name = item[0])
		end_node = Node("Entity", name = item[1])
		relation = Relationship(start_node, item[2], end_node)
		g.merge(start_node, "Entity", "name")
		g.merge(end_node, "Entity", "name")
		g.merge(relation, "Entity", "name")


