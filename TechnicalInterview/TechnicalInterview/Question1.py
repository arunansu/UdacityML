#Question 1: Given two strings s and t, determine whether some anagram of t is a substring of s. 
#For example: if s = "udacity" and t = "ad", then the function returns True. 
#Your function definition should look like: question1(s, t), and return a boolean True or False.

letters = {'a': 2, 'b': 3, 'c': 5, 'd': 7, 'e': 11, 'f': 13, 'g': 17, 'h': 19, 'i': 23, 'j': 29, 'k': 31, 
                'l': 37, 'm': 41, 'n': 43, 'o': 47, 'p': 53, 'q': 59, 'r': 61, 's': 67, 't': 71, 'u': 73, 'v': 79, 
                'w': 83, 'x': 89, 'y': 97, 'z': 101}

def hash(s):
    hashValue = 0
    for c in s.lower():
        if(c in letters):
            hashValue += letters[c]
        else:
            return 0
    return hashValue

def question1(s, t):
    if ((s != None) and (t != None) and (len(s) != 0) and (len(t) != 0) and (len(s) >= len(t))):
        i = 0
        t_hash = hash(t)
        if(t_hash == 0):
            return False
        while((i + len(t)) < len(s)):
            s_hash = hash(s[i:i + len(t)])
            if(s_hash == 0):
                return False
            if(s_hash == t_hash):
                return True
            i += 1
    return False

print question1(None, 'a')
print question1('a', None)
print question1(None, None)
print question1('', '')
print question1('', 'a')
print question1('a', '')
print question1('abcd', 'abcde')
print question1("Udacty", "AD")
print question1("I am Sam", "Sam")
print question1("abc@#$", "@#$")
print question1("abc", "def")