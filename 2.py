# 1

myString = "01234567"
for i in range(len(myString)):

    if i < len(myString)/2:
        print(myString[i:i+3:2])
'''
#2

myString = "01234567"
i=0
while i<len(myString):
    if i < len(myString)/2:
        print(myString[i:i+2])
    i+=1


#3
myString = "01234567"
j=0
for i in range(len(myString)):

    if i < len(myString)/2:
        print(myString[i:i+3:2])
    elif j==0:
        print('STOP')
        j=1

#2b

k = ["milk","eggs","bread","cheese","jam"]
v = [1,12,2,5,2]
def f(keys,vals):
    d={}
    for i in range(len(keys)):
        d[keys[i]] = vals[i]
    return(d)
D1 = f(k,v)
print(D1)

#print(D1['yoghurt'])
# key error -- yoghurt not present
k = ["milk","eggs","bread","cheese","jam"]
v = [1,12,2,5,2]
def f(keys,vals):
    d={}
    for i in range(len(keys)-1,-1,-1):
        d[keys[i]] = vals[i]
    return(d)
D2 = f(k,v)

print(D2)

#for i in range(len(keys)-1,-1,-1):

'''

#2 last tuple wala

k = ["milk","eggs","bread","cheese","jam"]
v = [1,12,2,5,2]
def f(keys,vals):
    d=[]
    for i in range(len(keys)):
        d.append((keys[i],vals[i]))
    return d

print(f(k,v))
