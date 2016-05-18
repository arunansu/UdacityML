#Question 4: Find the least common ancestor between two nodes on a binary search tree. 
#The least common ancestor is the farthest node from the root that is an ancestor of both nodes. 
#For example, the root is a common ancestor of all nodes on the tree, 
#but if both nodes are descendents of the root's left child, 
#then that left child might be the lowest common ancestor. 
#You can assume that both nodes are in the tree, and the tree itself adheres to all BST properties. 
#The function definition should look like "question4(T, r, n1, n2)", 
#where T is the tree represented as a matrix, where the index of the list is equal to the integer 
#stored in that node and a 1 represents a child node, r is a non-negative integer representing the root, 
#and n1 and n2 are non-negative integers representing the two nodes in no particular order. 
#For example, one test case might be 
#question4([[0,1,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,1],[0,0,0,0,0]],3,1,4), and the answer would be 3.

class Node(object):
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def populateTree(T,r,root):
    i = 0
    for column in T[r]:
        if column == 1 and i < root.value:
            root.left = Node(i)
        elif column == 1 and i > root.value:
            root.right = Node(i)
        i += 1
    if root.left != None:
        populateTree(T, root.left.value, root.left)
    if root.right != None:
        populateTree(T, root.right.value, root.right)

def LCA(root, n1, n2):
    if root.value == None:
        return None
    elif root.value > n1 and root.value > n2:
        return LCA(root.left, n1, n2)
    elif root.value <= n1 and root.value < n2:
        return LCA(root.right, n1, n2)
    return root.value

def question4(T, r, n1, n2):
    root = Node(r)
    populateTree(T, r, root)
    return LCA(root, n1, n2)

print question4([[0,1,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[1,0,0,0,1],[0,0,0,0,0]],3,1,4)  #3
print question4([[0,0,0,0,0],[1,0,1,0,0],[0,0,0,0,0],[0,1,0,0,1],[0,0,0,0,0]],3,0,2) #1
print question4([[0,0,0,0,0],[1,0,0,0,0],[0,1,0,1,0],[0,0,0,0,0],[0,0,1,0,0]],4,0,3) #2