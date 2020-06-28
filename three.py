def createAirportTimes(tuplst):
    adict = {}
    for el in tuplst:
        adict[el[0]]= el[1]
    return adict

tu = [('DUB',0.8),('FRA',1.5),('JFK',6.5)]

ws=createAirportTimes(tu)

def printAirportTimes(ws):
    ls1 =[]
    ls2 =[]
    lst=[]
    for i in ws.keys():
        ls1.append(i)
    for j in ws.values():
        ls2.append(j)
    for i in range(len(ls1)):
        lst.append((ls1[i],ls2[i]))
    for i in range(len(lst)):
        print(lst[i])


def second_shortest(adict):
    lst1=[]
    for i in adict.values():
        lst1.append(i)
    sortlist = sorted(lst1)
    val = sortlist[1]
    dd = adict[val]
    return(dd,val)

def second_shortest(ad):
    val = list(ad.values())
    key = list(ad.keys())
    val_s = sorted(val)[1]
    index = val.index(val_s)
    key_s = key[index]
    second_shortest = (key_s,val_s)
    return second_shortest

