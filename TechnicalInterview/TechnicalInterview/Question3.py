#Question 3: Given an undirected graph G, find the minimum spanning tree within G. 
#A minimum spanning tree connects all vertices in a graph with the smallest possible total weight of edges. 
#Your function should take in and return an adjacency list structured like this: 
#{'A':[('B',2)],'B':[('A',2),('C',5)],'C':[('B',5)]}. Vertices are represented as unique strings. 
#The function definition should be "question3(G)"
import sys

def question3(G):
    mintree = {}        
    mindistance = sys.maxint
    mintuple = (0,0)
    for node in G:
        mintree[node] = []
        for edge in G[node]:
            if edge[0] not in mintree:
                if edge[1] < mindistance:
                    mintuple = (edge[0], edge[1])
                    mindistance = edge[1]
            else:
                mintree[node].append((edge[0], edge[1]))
        mintree[node].append(mintuple)
        mindistance = sys.maxint
    return mintree

mintree = question3({'A':[('B',2)],'B':[('A',2),('C',5)],'C':[('B',5)]})
for node in mintree:
    print node
    for edge in  mintree[node]:
        print edge[0], edge[1]
        