# coding:utf-8
import numpy as np
from cs import norm


def main():
    N = 9
    M = 10
    K = 2

    A = np.random.random([N, M])
    x = np.zeros(M)
    x[np.random.randint(0, M, K)] = 1
    y = np.dot(A, x)

    print('x', x)
    print('MP', norm.MP(y, A, K)['x'])
    print('OMP', norm.gOMP(y, A, K)['x'])
    print('gOMP', norm.gOMP(y, A, K)['x'])
    print('ROMP', norm.ROMP(y, A, K)['x'])
    print('CoSaMP', norm.CoSaMP(y, A, K)['x'])
    print('stOMP', norm.stOMP(y, A)['x'])
    print('SWOMP', norm.SWOMP(y, A)['x'])
    print('SAMP', norm.SAMP(y, A)['x'])
        

if __name__ == '__main__':
    main()
