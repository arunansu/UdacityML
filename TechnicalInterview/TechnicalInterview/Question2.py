#Given a string a, find the longest palindromic substring contained in a. Your function definition 
#should look like "question2(a)", and return a string.

def question2(a):
    palindrome = ""
    len_a = len(a)
    if len_a > 0:
        palindrome = a[0]
    for i in xrange(len_a):
        for j in xrange(i):
            substring = a[j:i + 1]
            if substring == substring[::-1]:
                if len(substring) > len(palindrome):
                    palindrome = substring
    return palindrome

print question2("abafgdfgfjracecar") #"racecar"
print question2("abcdefg") #""
print question2("Anna is taking the kayak to the river") #" kayak " 