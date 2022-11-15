
import csv
import numpy as np
import time
import multiprocessing as mp
import sys
from scipy import optimize, integrate
import warnings

logeps = np.logspace(-7, 1, 100)

def bound(input):
    [n, f] = input
    renorm = 1
    pdf = lambda eps: np.exp(-np.sum(3 * np.abs(Z[n][f]) ** 2 / (3 + eps ** 2 * S[n][f] ** 2))) / np.prod(renorm * (3 + eps ** 2 * S[n][f] ** 2)) * np.sqrt(np.sum(eps ** 2 * S[n][f] ** 4 / (3 + eps ** 2 * S[n][f] ** 2) ** 2))
    pdfvals = np.array([pdf(x) for x in logeps])
    while np.all(pdfvals == 0):
        renorm *= 0.1
        pdf = lambda eps: np.exp(-np.sum(3 * np.abs(Z[n][f]) ** 2 / (3 + eps ** 2 * S[n][f] ** 2))) / np.prod(renorm * (3 + eps ** 2 * S[n][f] ** 2)) * np.sqrt(np.sum(eps ** 2 * S[n][f] ** 4 / (3 + eps ** 2 * S[n][f] ** 2) ** 2))
        pdfvals = np.array([pdf(x) for x in logeps])
    pdfmax = np.amax(pdfvals)
    pdfargmax = np.argmax(pdfvals)
    if logeps[pdfargmax] >= 1: return(1)
    tail = np.nonzero(pdfvals[pdfargmax:] / pdfmax <= 1e-5)[0] + pdfargmax
    if len(tail) != 0:
        upper = logeps[tail[0]]
    else:
        return(1)
    norm = integrate.quad(pdf, 0, upper)[0]

    normedpdf = lambda eps: pdf(eps) / norm
    bound = optimize.fsolve(lambda eps: integrate.quad(normedpdf, 0, eps)[0] - 0.95, upper / 2)[0]
    count = 0
    direction = -1
    while abs(integrate.quad(normedpdf, 0, bound)[0] - 0.95) >= 1e-5 and count < 5:
        bound = optimize.fsolve(lambda eps: integrate.quad(normedpdf, 0, eps)[0] - 0.95, 0.9 ** (direction * (count + 1)) * upper / 2)[0]
        if direction == 1: count += 1
        direction *= -1
    if abs(integrate.quad(normedpdf, 0, bound)[0] - 0.95) >= 1e-5:
        print(str(n) + ' ' + str(f) + ' bad solve bound=' + str(bound) + ' int=' + str(integrate.quad(normedpdf, 0, bound)))
        sys.stdout.flush()
    if f % 10000 == 0:
        print(str(n) + ' ' + str(f) + ' end ' + str(time.time() - t0))
        sys.stdout.flush()
    return(min(bound, 1))