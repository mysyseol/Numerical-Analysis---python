# my_module.py
# Applied Numerical Methods for Engineers and Scientists, 3rd ed.
# Steven C. Chapra

import numpy as np

# chap 5. Roots: Bracketing Methods
def bisection(fun,x0,x1,tol=1e-3):
    f0 = fun(x0)
    # f1 = fun(x1)
    k = 0
    while 1:
        k += 1
        x2 = (x0 + x1)/2
        f2 = fun(x2)
        if f0*f2 > 0:
            x0, f0 = x2, f2
        else:
            x1 = x2
        if abs(x1-x0) < tol:
            return k, x1

def secant(fun,x0,x1,tol=1e-3):
    f0 = fun(x0)
    f1 = fun(x1)
    k = 0
    while 1:
        k += 1
        x2 = x1 - f1*((x1 - x0)/(f1 - f0))
        x0 = x1
        x1 = x2
        if abs(x1-x0) < tol:
            return k, x1
        f0 = f1
        f1 = fun(x2)

# chap 6. Roots: Open Methods
def fixed_point_iteration(fun,x,tol=1e-3):
    k = 0
    while 1:
        k += 1
        xold = x
        x = fun(x)
        if abs(x-xold) < tol:
            return k, x

def newton_raphson(fun,dfun,x, tol=1e-3):
    k = 0
    while 1:
        k += 1
        xold = x
        f = fun(x)
        df = dfun(x)
        x = x - f/df
        if abs(x - xold) < tol:
            return k, x

# chap 7. Optimization

def golden_section_search(fun,x1,x4,tol=1e-3):
    p = 1.0 - 0.61803398874989
    d = (x4-x1)*p
    x2, x3 = x1+d, x4-d
    f2, f3 = fun(x2), fun(x3)
    k = 0
    while 1:
        k += 1
        if f2 < f3:
            x4, x3, f3 = x3, x2, f2
            x2 = x1 + (x4-x1)*p
            f2 = fun(x2)
        else:
            x1, x2, f2 = x2, x3, f3
            x3 = x4 - (x4-x1)*p
            f3 = fun(x3)
        if abs(x2-x3) < tol:
            return k, (x2+x3)/2

# chap 9. Gauss Elimination

def gauss_elimination(a,b):
    # naive gauss elimination
    m, n = a.shape
    # forward elimination
    for k in range(n-1):
        for i in range(k+1,n):
            factor = a[i,k]/a[k,k]
            a[i,k:n] = a[i,k:n] - factor*a[k,k:n]
            b[i] = b[i] - factor*b[k]
    # backward substitution
    x = np.zeros((n,1))
    x[-1] = b[-1]/a[-1,-1]
    for k in range(n-2,-1,-1):
        # x[k] = (b[k] - np.matmul(a[k,k+1:],x[k+1:]))/a[k,k]
        x[k] = (b[k] - a[k,k+1:].dot(x[k+1:]))/a[k,k]
        # x[k] = (b[k] - np.dot(a[k,k+1:], x[k+1:]))/a[k,k]
    return x

def tdma(tri):
    # tri-diagonal system
    e = tri[:,0]
    f = tri[:,1]
    g = tri[:,2]
    r = tri[:,3]
    n = len(f)
    for k in range(1,n):
        factor = e[k]/f[k-1]
        f[k] = f[k] - factor*g[k-1]
        r[k] = r[k] - factor*r[k-1]
    x = np.zeros((n,1))
    x[-1] = r[-1]/f[-1]
    for k in range(n-2,-1,-1):
        x[k] = (r[k] - g[k]*x[k+1])/f[k]
    return x

# chap 10. LU factorization
def lu_factorization(a):
    # naive gauss elimination
    m, n = a.shape
    # forward elimination
    for k in range(n-1):
        for i in range(k+1,n):
            factor = a[i,k]/a[k,k]
            a[i,k:n] = a[i,k:n] - factor*a[k,k:n]
            a[i,k] = factor
    return a

def lu_solve(a,b):
    n = len(b)
    # forward substitution
    for i in range(n):
        b[i] -= np.matmul(a[i,:i],b[:i])
    # backward substitution
    b[n-1] = b[n-1]/a[n-1,n-1]
    for i in range(n-2,-1,-1):
        # b[i] = (b[i] - a[i,i+1:].dot(b[i+1:]))/a[i,i]
        b[i] = (b[i] - np.dot(a[i,i+1:],b[i+1:]))/a[i,i]
    return b

# def cholesky_factorization(a):
#     # a = u.T * u
#     m, n = a.shape

# chap 11. matrix inverse
def matrix_inverse(a):
    m, n = a.shape
    b = np.eye(n)
    a = lu_factorization(a)
    for i in range(n):
        b[:,i] = lu_solve(a,b[:,i])
    return b

# chap 12. gauss seidel iteration
def gauss_seidel(a,b,tol=1e-3):
    """
    gauss seidel iteration to solve a*x=b

    """
    m, n = a.shape
    x = np.ones((n,1))
    k = 0
    while 1:
        k += 1
        for i in range(n):
            s = 0
            for j in range(n):
                if i == j:
                    continue
                s += a[i,j]*x[j]
            x[i] = (b[i]-s)/a[i,i]
        if np.linalg.norm(np.matmul(a,x)-b) < tol: break
    return x, k

# def fun_system(x):
#     x1 = x[0]
#     x2 = x[1]
#     f = np.array([
#         x1**2 + x1*x2 - 10,
#         x2 + 3*x1*x2**2 - 57
#         ])
#     return f
#
# def jacobian(fun,x):
#     n = len(x)
#     jacob = np.zeros((n,n))
#     delta = 0.1
#     d = np.eye(n)
#     f = fun(x)
#     for i in range(n):
#         x1 = x + delta*d[:,i]
#         f1 = fun(x1)
#         jacob[:,i] = (f1 - f)/delta
#     return jacob
#
# def newton_raphson_system(fun,x):

# chap 19. numerical integration

def trapz(y, x=None, dx=1.0):
    """
    Integrate using the composite trapezoidal rule.
    ed. 2017-12-20
    """
    y = np.asanyarray(y)
    if x is None:
        return dx/2.*(y[0]+y[-1]+2.*np.sum(y[1:-1]))
    else:
        x = np.asanyarray(x)
        return np.sum((x[1:]-x[:-1])*(y[1:]+y[:-1])/2)

def simps(y, dx=1.0):
    """
    Integrate using the composite simpson rule.
    number of data point must be odd
    ed. 2017-12-20
    """
    y = np.asanyarray(y)
    return dx/3.*(y[0]+y[-1]+4.*np.sum(y[1:-1:2])+2.*np.sum(y[2:-2:2]))

def quad_gauss(fun,a,b):
    # ng = 3
    xg = np.array([-0.7745966692414834, 0.0000000000000000, 0.7745966692414834])
    wg = np.array([ 0.5555555555555556, 0.8888888888888888, 0.5555555555555556])
    x = ((b+a)+(b-a)*xg)/2.
    f = fun(x)
    res = np.sum(wg*f)*(b-a)/2.
    return res

# p = [400., -900., 675., -200, 25., 0.2]
# fun = lambda x: np.polyval(p,x)
# I = quad_gauss(fun,0,0.8)
# print(I)


def quad_adapt(fun, a, b, tol=1e-3):
    c = (a + b)/2.
    fa, fb, fc = fun(a), fun(b), fun(c)
    q = quad_step(fun, a, b, tol, fa, fc, fb)
    return q

def quad_step(fun,a,b,tol,fa,fc,fb):
    """
    a d c e b
    """
    h, c = b - a, (a + b)/2.
    fd = fun((a + c)/2.)
    fe = fun((c + b)/2.)
    q1 = h/6.*(fa + 4.*fc + fb)
    q2 = h/12. *(fa + 4.*fd + 2.*fc + 4.*fe + fb)
    if np.absolute(q2 - q1) < tol:
        q = q2 + (q2 - q1)/15.
    else:
        qa = quad_step(fun, a, c, tol, fa, fd, fc)
        qb = quad_step(fun, c, b, tol, fc, fe, fb)
        q = qa + qb
    return q

# chap 22. ODE, ordinary differential equation

def ode_euler(fun,y0,t):
    n = len(t)
    y = np.ones(t.shape)*y0
    for i in range(1,n):
        h = t[i]-t[i-1]
        y[i] = y[i-1] + fun(t[i-1], y[i-1])*h
    return y

def ode_heun(fun,y0,t,tol=1e-3):
    n = len(t)
    y = np.ones(t.shape)*y0
    for i in range(1,n):
        t1 = t[i-1]
        t2 = t[i]
        h = t2-t1
        y1 = y[i-1]
        f1 = fun(t1, y1)
        # prediction
        y2 = y1 + f1*h
        k = 0
        while 1:
            k += 1
            y2old = y2
            f2 = fun(t2, y2)
            f12 = (f1 + f2)/2.
            # correction
            y2 = y1 + f12*h
            if np.absolute(y2-y2old) < tol:
                y[i] = y2
                break
    return y

def ode_rk4(fun,y0,t):
    n = len(t)
    y = np.ones(t.shape)*y0
    for i in range(1,n):
        t1, h, y1 = t[i-1], t[i]-t[i-1], y[i-1]
        k1 = fun(t1, y1)
        k2 = fun(t1 + h/2., y1 + k1*h/2.)
        k3 = fun(t1 + h/2., y1 + k2*h/2.)
        k4 = fun(t1 + h, y1 + k3*h)
        y[i] = y1 + (k1 + 2.*k2 + 2.*k3 + k4)*h/6.
    return y

def ode_euler_sys(fun, y0, t):
    y0 = np.asanyarray(y0)
    t = np.asanyarray(t)
    n = len(t)
    m = len(y0)
    y = np.zeros((n,m))
    y[0] = y0

    for i in range(1,n):
        h = t[i]-t[i-1]
        y[i] = y[i-1] + fun(t[i-1], y[i-1])*h
    return y

def ode_rk4_sys(fun, y0, t):
    y0 = np.asanyarray(y0)
    t = np.asanyarray(t)
    n = len(t)
    m = len(y0)
    y = np.zeros((n,m))
    y[0] = y0
    for i in range(1,n):
        t1, h, y1 = t[i-1], t[i]-t[i-1], y[i-1]
        k1 = fun(t1, y1)
        k2 = fun(t1 + h/2., y1 + k1*h/2.)
        k3 = fun(t1 + h/2., y1 + k2*h/2.)
        k4 = fun(t1 + h, y1 + k3*h)
        y[i] = y1 + (k1 + 2.*k2 + 2.*k3 + k4)*h/6.
    return y


if __name__ == "__main__":

#    def func_test(x):
#        return np.exp(-x)-x
#
#    def dfunc_test(x):
#        return -np.exp(-x)-1
#
#    k, x = bisection(func_test, 0., 1.)
#    print("bisection method\nk = %5d, x = %9.4f " %(k, x) )
#
#    k, x = secant(func_test, 0., 1.)
#    print("secant method\nk = %5d, x = %9.4f " %(k, x) )
#
#    def g_fun(x): return np.exp(-x)
#    k, x = fixed_point_iteration(g_fun, 0.5)
#    print("fixed_point_iteration\nk = %5d, x = %9.4f " %(k, x) )
#
#    k, x = newton_raphson(func_test,dfunc_test, 0.5)
#    print("newton_raphson method\nk = %5d, x = %9.4f " %(k, x) )

#    def func_test2(x):
#        f= (x**2)/10. - 2.*np.sin(x)
#        return f
#   
#    k, x = golden_section_search(func_test2, 0., 4.)
#    print("golden_section_search\nk = %5d, x = %9.4f " %(k, x) )

     A = np.array([[3, -0.1, -0.2],[0.1, 7., -0.3],[0.3, -0.2, 10.]])
     b = np.array([[7.85],[-19.3],[71.4]])
     x = gauss_elimination(A,b)
     print(x)

    # tri = np.array([
    # [-1, 2.04, -1,  40.8],
    # [-1, 2.04, -1,   0.8],
    # [-1, 2.04, -1,   0.8],
    # [-1, 2.04, -1, 200.8]
    # ])
    # print(tri)
    # x = tdma(tri)
    # print(x)

    # a = np.array([[3, -0.1, -0.2],[0.1, 7., -0.3],[0.3, -0.2, 10.]])
    # b = np.array([[7.85],[-19.3],[71.4]])
    # x = np.linalg.solve(a,b)
    # print(x)
    # x = gauss_elimination(a,b)
    # print(x)
    # x = gauss_seidel(a,b)
    # print(x[0])
    # print('iteration :', x[1])
    # a = lu_factorization(a)
    # x = lu_solve(a,b)
    # print(x)

    # a = np.array([[3, -0.1, -0.2],[0.1, 7., -0.3],[0.3, -0.2, 10.]])
    # ai = matrix_inverse(a)
    # print(ai)
    # res = np.matmul(a,ai)
    # print(res)

    # chapter 22, ODE solver

#    def odefun(t,y):
#        x, v = y
#        dydt = [v, 9.81 - 0.25/68.1*v**2]
#        return np.array(dydt)
#
#    t = np.linspace(0,10,21)
#    y0 = [0, 0]
#    y1 = ode_euler_sys(odefun,y0,t)
#    y2 = ode_rk4_sys(odefun,y0,t)
#    print(y1)
#    # y3 = ode_rk4(odefun,y0,t)
#
#    import matplotlib.pyplot as plt
#    plt.subplot(211)
#    plt.plot(t, y1[:,0], 'b.-', label='euler')
#    plt.plot(t, y2[:,0], 'r.-', label='rk4')
#    plt.legend(loc='best')
#    plt.xlabel('time, s')
#    plt.ylabel('x, m')
#    plt.grid()
#
#    plt.subplot(212)
#    plt.plot(t, y1[:,1], 'b.-', label='euler')
#    plt.plot(t, y2[:,1], 'r.-', label='rk4')
#    plt.legend(loc='best')
#    plt.xlabel('time, s')
#    plt.ylabel('v, m/s')
#    plt.grid()
#
#    plt.show()

    # print('my_module.py')
