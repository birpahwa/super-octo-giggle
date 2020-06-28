import sys

def rot(s,shift):
    result = ''
    numLetters = 26
    if shift >25 and shift <-25:
        print('Error in rot function - shift must be > -25 and < 25')
        sys.exit(-1)
    else:
        for v in s:
            c = ord(v)
            if c >64 and c <91:
                ss= c + shift
                if ss >90:
                    temp = ss-90
                    c= 64+temp
                else:
                    c=ss
            if c > 96 and c <123:
                ss = c + shift
                if ss > 122:
                    temp = ss - 122
                    c = 96 + temp
                else:
                    c = ss

        # ord(v) - if v is a character this function returns
        #the Unicode integer that represents it
        # Unicode integers occur in the order ’a’, ’b’ and so on



        # Check if c represents a lower case or upper casecharacter
        # If it does then update value accordingly to
        #represent rotation.

         # chr(c) returns the character corresponding to theUnicode integer c
            result += chr(c)
        return(result)


print(rot('Viky',5))