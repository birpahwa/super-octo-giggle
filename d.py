lst=[0.0,3.0,6.0]
def meanVec(lst):
    return(sum(lst)/len(lst))

def varVec(lst):
    va=[]
    for i in lst:
        va.append((i-meanVec(lst))**2)

    return(sum(va)/len(lst))

print(meanVec(lst))
print(varVec(lst))