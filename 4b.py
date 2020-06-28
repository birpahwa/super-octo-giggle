def Fn(f,g,n):
    if n == 0:
        return (0)
    elif n == 1:
        return (1)
    return(f(f,g,n-1) + g(f,g,n-2))

print(Fn(Fn,Fn,3))