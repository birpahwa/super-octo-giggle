'''
theProduct = 10
for i in range(1,30):
    print("i=",i)
    print("Running total =",theProduct)
    if i%3 == 0:
        theProduct *= i
print(theProduct)

# for loop
#3 to 6

def lOrder(l1,l2,l3):
    if len(l1)<len(l2) and len(l2)<len(l3):
        return True
    else:
        return False

l1 = [1]
l2=[2,3,4]
l3 =[5,3,4,6]

print(lOrder(l1,l2,l3))


def R3(str):
    return str[-1:-4:-1]

print(R3('fgjh'))

def catStrings(lst):
    dd=''
    for i in range(len(lst)):
        dd=dd+lst[i]
    return dd

print(catStrings(['my','name','is','bir']))


def returnUniqueVals(dict):
    li=[]
    for i in dict.values():
        li.append(i)
    uniset= set(li)
    return list(uniset)

print(returnUniqueVals({'ss':3,'dd':4,'ff':4}))

'''


ages = {'N':1, 'R':2, 'W':0, 'J':4, 'H':1}
S = ['N','R']
for i in S:
    if i in ages:
        del(ages[i])

print(ages)