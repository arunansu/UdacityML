#Question 3: Given an undirected graph G, find the minimum spanning tree within G. 
#A minimum spanning tree connects all vertices in a graph with the smallest possible total weight of edges. 
#Your function should take in and return an adjacency list structured like this: 
#{'A':[('B',2)],'B':[('A',2),('C',5)],'C':[('B',5)]}. Vertices are represented as unique strings. 
#The function definition should be "question3(G)"
import sys

def question3(G):
    mintree = {}        
    mindistance = sys.maxint
    mintuple = (0, 0)
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

print question3({'A':[('B',2)],'B':[('A',2),('C',5)],'C':[('B',5)]})
#{'A': [('B', 2)], 'C': [('B', 5)], 'B': [('A', 2), ('C', 5), ('B', 5)]}
print question3({'A':[('B',7),('C',9),('F',14)],'B':[('A',7),('C',10),('D',15)],'C':[('A',9),('B',10),('D',11),('F',2)],
                    'D':[('B',15),('E',6),('C',11)],'E':[('D',6),('F',9)],'F':[('E',9),('C',2),('A',14)]})
#{'A': [('B', 7)], 'C': [('A', 9), ('F', 2)], 'B': [('A', 7), ('C', 10), ('D', 15)], 
#'E': [('D', 6)], 'D': [('B', 15), ('E', 6), ('C', 11), ('D', 6)], 'F': [('E', 9), ('C', 2), ('A', 14), ('D', 6)]}
print question3({'A':[('D',9),('E',1)],'B':[('A',5),('C',3),('D',1),('E',2)],'C':[('B',3),('D',14),('E',7)],
                'D':[('A',9),('B',1),('C',14)],'E':[('A',1),('B',2),('C',7)]})
#{'A': [('E', 1)], 'C': [('B', 3)], 'B': [('A', 5), ('C', 3), ('D', 1)], 'E': [('A', 1), ('B', 2), ('C', 7), ('D', 1)], 
#'D': [('A', 9), ('B', 1), ('C', 14), ('D', 1)]}