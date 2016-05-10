#Given a string a, find the longest palindromic substring contained in a. Your function definition 
#should look like "question2(a)", and return a string.

def question2(a):
    palindrom = ""
    for i in xrange(len(a)):
        for j in xrange(i):
            substring = a[j:i + 1]
            if substring == substring[::-1]:
                if len(substring) > len(palindrom):
                    palindrom = substring
    return palindrom

print question2("abafgdfgfjracecar")
print question2("abcdefg")