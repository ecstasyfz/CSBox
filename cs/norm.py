# coding: utf-8
"""
    Author: ecstasyfz

    Summary of methods:
    - l0
        - MP
        - OMP
        - gOMP
        - ROMP
        - CoSaMP
        - SWOMP
        - stOMP
        - SAMP
    - l1
"""
import numpy as np
import math


# ---L_0---
def MP(y, A, K=None, eps=None):
    if K is None:
        K = A.shape[1]
    if eps is None:
        eps = np.finfo(np.float).eps *10

    k = 0
    r = y

    I = np.array([], dtype=np.int64)
    B = A / [np.linalg.norm(row) for row in A.T]
    x = np.zeros(A.shape[1])

    while k < K and np.linalg.norm(r) > eps:
        u = [np.inner(r, b) for b in B.T]
        j = np.abs(u).argmax()
        I = np.append(I, [j])
        r = r - u[j] * B.T[j]
        x[j] = x[j] + u[j] / np.linalg.norm(A.T[j])

        k = k + 1

    return {'I': I, 'x': x}


def OMP(y, A, K=None, eps=None):
    if K is None:
        K = A.shape[1]
    if eps is None:
        eps = np.finfo(np.float).eps *10

    k = 0
    r = y

    B = A / [np.linalg.norm(row) for row in A.T]
    I = np.array([], dtype=np.int64)

    while k < K and np.linalg.norm(r) > eps:
        u = np.abs([np.inner(r, b) for b in B.T])
        j = u.argmax()
        I = np.union1d(I, [j])

        r = y - np.dot(A[:, I], np.linalg.lstsq(A[:, I], y)[0])
        k = k + 1

    theta = np.linalg.lstsq(A[:, I], y)[0]
    x = np.zeros(A.shape[1])
    for i, v in zip(I, theta):
        x[i] = v

    return {'I': I, 'x': x}


def gOMP(y, A, K=None, S=None, eps=None):
    if K is None:
        K = A.shape[1]
    if S is None:
        S = K // 4 > 0 and K / 4 or 1
    if eps is None:
        eps = np.finfo(np.float).eps *10

    k = 0
    r = y

    B = A / [np.linalg.norm(row) for row in A.T]
    I = np.array([], dtype=np.int64)

    while k < K and np.linalg.norm(r) > eps:
        u = np.abs([np.inner(r, b) for b in B.T])
        J = u.argsort()[-S:]
        I = np.union1d(I, J)

        r = y - np.dot(A[:, I], np.linalg.lstsq(A[:, I], y)[0])
        k = k + 1

    theta = np.linalg.lstsq(A[:, I], y)[0]
    x = np.zeros(A.shape[1])
    for i, v in zip(I, theta):
        x[i] = v

    return {'I': I, 'x': x}


def ROMP(y, A, K=None, eps=None):
    if K is None:
        K = A.shape[1]
    if eps is None:
        eps = np.finfo(np.float).eps *10

    k = 0
    r = y

    B = A / [np.linalg.norm(row) for row in A.T]
    I = np.array([], dtype=np.int64)

    while k < K and np.linalg.norm(r) > eps and len(I) < K * 2:
        u = np.abs([np.inner(r, b) for b in B.T])
        J = [i for i in u.argsort()[-K:] if u[i] != 0][::-1]

        max_energe = 0
        J_0 = []
        for hi in range(len(J)):
            inner_limit = u[hi] / 2
            lo = hi + 1
            while lo < len(J):
                if u[lo] < inner_limit:
                    break
                lo += 1
            energy = np.sum([np.linalg.norm(B.T[i]) for i in J[hi: lo]])
            if energy > max_energe:
                max_energe = energy
                J_0 = J[hi: lo]

        I = np.union1d(I, J_0)

        r = y - np.dot(A[:, I], np.linalg.lstsq(A[:, I], y)[0])
        k = k + 1

    theta = np.linalg.lstsq(A[:, I], y)[0]
    x = np.zeros(A.shape[1])
    for i, v in zip(I, theta):
        x[i] = v

    return {'I': I, 'x': x}


def CoSaMP(y, A, K=None, S=None, eps=None):
    if K is None:
        K = A.shape[1]
    if S is None:
        S = K
    if eps is None:
        eps = np.finfo(np.float).eps *10

    t = 0
    r = y

    B = A / [np.linalg.norm(row) for row in A.T]
    I = np.array([], dtype=np.int64)

    while t < S and np.linalg.norm(r) > eps:
        u = np.abs([np.inner(r, b) for b in B.T])
        J = np.array([i for i in u.argsort()[-K * 2:] if u[i] != 0])
        I = np.union1d(I, J)

        theta = np.linalg.lstsq(A[:, I], y)[0]
        I = I[theta.argsort()[-K:]]

        r = y - np.dot(A[:, I], np.sort(theta)[-K:])
        t = t + 1

    theta = np.linalg.lstsq(A[:, I], y)[0]
    x = np.zeros(A.shape[1])
    for i, v in zip(I, theta):
        x[i] = v

    return {'I': I, 'x': x}


def stOMP(y, A, S=None, ts=None, eps=None):
    if S is None:
        S = 10
    if ts is None:
        ts = 2.5
    if eps is None:
        eps = np.finfo(np.float).eps *10

    t = 0
    r = y

    B = A / [np.linalg.norm(row) for row in A.T]
    I = np.array([], dtype=np.int64)

    while t < S and np.linalg.norm(r) > eps:
        u = np.abs([np.inner(r, b) for b in B.T])
        J = np.array([i for i in range(len(u)) if u[i] >= ts *
                      np.linalg.norm(r) / math.sqrt(A.shape[1])], dtype=np.int64)

        I_prev = I
        I = np.union1d(I, J)
        if len(I_prev) == len(I):
            break

        r = y - np.dot(A[:, I], np.linalg.lstsq(A[:, I], y)[0])
        t = t + 1

    theta = np.linalg.lstsq(A[:, I], y)[0]
    x = np.zeros(A.shape[1])
    for i, v in zip(I, theta):
        x[i] = v

    return {'I': I, 'x': x}


def SWOMP(y, A, S=None, alpha=None, eps=None):
    if S is None:
        S = 10
    if alpha is None:
        alpha = 0.5
    if eps is None:
        eps = np.finfo(np.float).eps *10

    t = 0
    r = y

    B = A / [np.linalg.norm(row) for row in A.T]
    I = np.array([], dtype=np.int64)

    while t < S and np.linalg.norm(r) > eps:
        u = np.abs([np.inner(r, b) for b in B.T])
        J = np.array([i for i in range(len(u)) if u[i]
                      >= alpha * max(u)], dtype=np.int64)

        I_prev = I
        I = np.union1d(I, J)
        if len(I_prev) == len(I):
            break

        r = y - np.dot(A[:, I], np.linalg.lstsq(A[:, I], y)[0])
        t = t + 1

    theta = np.linalg.lstsq(A[:, I], y)[0]
    x = np.zeros(A.shape[1])
    for i, v in zip(I, theta):
        x[i] = v

    return {'I': I, 'x': x}


def SAMP(y, A, S=None, eps=None):
    if S is None:
        S = 1
    if eps is None:
        eps = np.finfo(np.float).eps *10

    t = 0
    r = y
    L = S

    B = A / [np.linalg.norm(row) for row in A.T]
    I = np.array([], dtype=np.int64)

    while t < A.shape[1]:
        u = np.abs([np.inner(r, b) for b in B.T])
        J = u.argsort()[-L:][::-1]
        C = np.union1d(I, J)

        theta = np.linalg.lstsq(A[:, C], y)[0]
        F = C[theta.argsort()[-L:]]

        r_new = y - np.dot(A[:, F], np.sort(theta)[-L:])

        r_new_norm = np.linalg.norm(r_new)
        if r_new_norm <= eps:
            I = F
            break
        elif r_new_norm >= np.linalg.norm(r):
            L = L + S
            continue
        else:
            I = F
            r = r_new
            t = t + 1

    theta = np.linalg.lstsq(A[:, I], y)[0]
    x = np.zeros(A.shape[1])
    for i, v in zip(I, theta):
        x[i] = v

    return {'I': I, 'x': x}
