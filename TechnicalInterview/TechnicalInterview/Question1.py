#Question 1: Given two strings s and t, determine whether some anagram of t is a substring of s. 
#For example: if s = "udacity" and t = "ad", then the function returns True. 
#Your function definition should look like: question1(s, t), and return a boolean True or False.

def hash(s):
    hash = {}
    for c in s:
        if c in hash:
            hash[c] = hash[c] + 1
        else:
            hash[c] = 1
    return hash

def matchHash(s, t):
    for c in s:
        if c not in t:
            return False
        else:
            if s[c] != t[c]:
                return False
    return True

def question1(s, t):
    if s == None or t == None:
        return False
    if s == t:
        return True
    s_len = len(s)
    t_len = len(t)
    if s_len >= t_len and t_len != 0:
        hash_t = hash(t)
        for i in xrange(s_len):
            if i + t_len <= s_len:
                hash_s = hash(s[i:i + t_len])
                if matchHash(hash_s, hash_t):
                    return True
    return False

print question1(None, 'a') #False
print question1('a', None) #False
print question1(None, None) #False
print question1('', '') #True
print question1('', 'a') #False
print question1('a', '') #False
print question1('abcd', 'abcde') #False
print question1("Udacty", "AD") #False
print question1("I am Sam", "Sam") #True
print question1("abc@#$", "@#$") #True
print question1("abc", "def") #False