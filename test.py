lst = ['ss', 'dd', 'ff', 'gg']


def R2(lst):
    """

    :type lst: taking reverse of first two elements of list
    """
    lst[0], lst[1] = lst[1], lst[0]
    return lst


print(R2(lst))

tuplst = [('aa', 'ss'), ('ff', 'dd'), ('gg', 'jj')]


def catTup(tuplst):
    '''

    :param tuplst:
    :return: frst element of all tuples in a list
    '''
    flist = []
    for element in tuplst:
        flist.append(element[0])
    return flist


print(catTup(tuplst))

adict = {'ss': 4, 'dd': 7, 'rr': 5}


def getter(adict, k):
    for el in adict.keys():

        if k in adict.keys():
            return adict[k]
        else:
            return None


print(getter(adict, 'dd'))