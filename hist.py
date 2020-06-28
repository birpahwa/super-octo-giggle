def histogram(str):
    adict={}
    for e in str:
        if e in adict:
            adict[e] +=1
        else:
            adict[e]=1
    return adict

str='Parrot'

print(histogram(str))


def filter_even(tup):
    tu = tup[1:len(tup):2]
    return (tu)


tup =('one','two','three','four','five')


print(filter_even(tup))

def reverse(li):
    return(li[-1:-len(li)-1:-1])


li = [0,1,2]

print(reverse(li))