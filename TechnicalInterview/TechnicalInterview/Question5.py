#Question 5: Find the element in a singly linked list that's m elements from the end. 
#For example, if a linked list has 5 elements, the 3rd element from the end is the 3rd element. 
#The function definition should look like "question5(ll, m)", 
#where ll is the first node of a linked list and m is the "mth number from the end". 

class Node(object):
  def __init__(self, data):
    self.data = data
    self.next = None

def question5(ll, m):
    i = 2
    p1 = ll
    p2 = ll.next
    while p2.next:
        i += 1
        p2 = p2.next
        if i > m:
            p1 = p1.next
    return p1

ll = Node(1)
ll.next = Node(2)
ll.next.next = Node(3)
ll.next.next.next = Node(4)
ll.next.next.next.next = Node(5)

print question5(ll, 3).data #Should be 3
print question5(ll, 4).data #Should be 2
print question5(ll, 2).data #Should be 4
