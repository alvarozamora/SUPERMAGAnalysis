import csv
import math
import numpy as np
import time
import multiprocessing as mp
import sys
import os
import psutil
import scipy.interpolate as interp

t0 = time.time()
process = psutil.Process(os.getpid())
filename = 'sz_timenetcdfmb4.csv'
print(filename)
print(process.memory_info().rss)

def maxprime(n):
    if n == 1: return(1)
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return(maxprime(int(n / i)))
    return(n)

skipyears = 38
years = 12#50 - skipyears
N = (365 * years + math.floor(years / 4)) * 24 * 60
N_cohT = (365 * 48 + math.floor(48 / 4)) * 24 * 60
p = 0.03
cohTs = int(np.around(np.log(N_cohT / 1e6) / (2 * np.log(1 + p)))) + 1
cohT = [int(np.around(N_cohT / (1 + p) ** (2 * n))) for n in range(cohTs)]
for n in range(1, cohTs):
    cohT[n] += np.argmin([maxprime(T) for T in range(cohT[n] - 10, cohT[n] + 11)]) - 10
#print(cohT)
#print([maxprime(T) for T in cohT])
lower = int(np.around(1e6 / (1 + p)))
freqbounds = [[lower, int(np.ceil(cohT[n] * lower / cohT[n + 1]))] for n in range(cohTs - 1)]
freqbounds[0][0] = 0
freqbounds.append([lower, cohT[-1]])
freqs = [np.arange(freqbounds[n][0], freqbounds[n][1]) / (60 * cohT[n]) for n in range(cohTs)]

uptime = 16384
downtime = 0
powerfreqs = np.linspace(0, 1 / 60, 2 * uptime + 1)

rho = 6.04e7
R = 0.0212751
fd = 1 / 86164
pad = lambda n: int(np.around(60 * cohT[n] * fd))

file = open('year_weighted_timeseries_netcdfmb.csv', newline = '')
reader = csv.reader(file, delimiter = ',', quotechar = '|')
timeseries = np.zeros((5, N))
for i in range(5):
    timeseries[i] = [float(x) for x in next(reader)[(365 * skipyears + math.floor((skipyears + 1) / 4)) * 24 * 60 : (365 * skipyears + math.floor((skipyears + 1) / 4)) * 24 * 60 + N]]
nans = np.nonzero(np.isnan(timeseries[0]))[0]
nanyears = 4 * np.floor(nans / 2103840).astype(int) + np.where(nans % 2103840 >= (365 + ((skipyears + 1) % 4 >= 3)) * 24 * 60, 1, 0) + np.where(nans % 2103840 >= (365 * 2 + ((skipyears + 1) % 4 >= 2)) * 24 * 60, 1, 0) + np.where(nans % 2103840 >= (365 * 3 + ((skipyears + 1) % 4 >= 1)) * 24 * 60, 1, 0)
#print(np.bincount(nanyears))
timeseries = np.nan_to_num(timeseries)
signal = np.zeros((7, N))
for i in range(7):
    signal[i] = [float(x) for x in next(reader)[(365 * skipyears + math.floor((skipyears + 1) / 4)) * 24 * 60 : (365 * skipyears + math.floor((skipyears + 1) / 4)) * 24 * 60 + N]]
signal = np.nan_to_num(signal)
file.close()
print(str(time.time() - t0) + ' ' + str(process.memory_info().rss))

muxfd = []
mux0 = []
muyfd = []
muy0 = []
muzfd = []
muz0 = []
# For every coherence time
for n in range(cohTs):

    # Find the number of chunks
    cohtimes = math.ceil(N / cohT[n])

    # Add 5 time (really frequency) series for each chunk
    muxfd.append(np.zeros((cohtimes, 5), dtype = complex))
    mux0.append(np.zeros((cohtimes, 5), dtype = complex))
    muyfd.append(np.zeros((cohtimes, 5), dtype = complex))
    muy0.append(np.zeros((cohtimes, 5), dtype = complex))
    muzfd.append(np.zeros((cohtimes, 5), dtype = complex))
    muz0.append(np.zeros((cohtimes, 5), dtype = complex))

    # Calculate the cis
    cis = np.cos(2 * math.pi * (pad(n) / cohT[n] - fd * 60) * np.arange(N)) + 1j * np.sin(2 * math.pi * (pad(n) / cohT[n] - fd * 60) * np.arange(N))

    # Auxilary trigonometric values
    cosfd = np.cos(2 * math.pi * fd * 60 * np.arange(N))
    sinfd = np.sin(2 * math.pi * fd * 60 * np.arange(N))
    cospad = np.cos(2 * math.pi * pad(n) / cohT[n] * np.arange(N))
    sinpad = np.sin(2 * math.pi * pad(n) / cohT[n] * np.arange(N))
#    cisf1 = np.cos(2 * math.pi * 2 * 0.0075 * 60 * np.arange(N)) + 1j * np.sin(2 * math.pi * 2 * 0.0075 * 60 * np.arange(N))
#    cispadf1 = np.cos(2 * math.pi * (2 * 0.0075 * 60 + pad(n) / cohT[n]) * np.arange(N)) + 1j * np.sin(2 * math.pi * (2 * 0.0075 * 60 + pad(n) / cohT[n]) * np.arange(N))
#    cispad_f1 = np.cos(2 * math.pi * (2 * 0.0075 * 60 - pad(n) / cohT[n]) * np.arange(N)) + 1j * np.sin(2 * math.pi * (2 * 0.0075 * 60 - pad(n) / cohT[n]) * np.arange(N))
#    cisf2 = np.cos(2 * math.pi * 2 * 0.001 * 60 * np.arange(N)) + 1j * np.sin(2 * math.pi * 2 * 0.001 * 60 * np.arange(N))
#    cispadf2 = np.cos(2 * math.pi * (2 * 0.001 * 60 + pad(n) / cohT[n]) * np.arange(N)) + 1j * np.sin(2 * math.pi * (2 * 0.001 * 60 + pad(n) / cohT[n]) * np.arange(N))
#    cispad_f2 = np.cos(2 * math.pi * (2 * 0.001 * 60 - pad(n) / cohT[n]) * np.arange(N)) + 1j * np.sin(2 * math.pi * (2 * 0.001 * 60 - pad(n) / cohT[n]) * np.arange(N))

    # Make sure that the nan values get replaced with zero
    cis[nans] = 0
    cosfd[nans] = 0
    sinfd[nans] = 0
    cospad[nans] = 0
    sinpad[nans] = 0
#    cisf1[nans] = 0
#    cispadf1[nans] = 0
#    cispad_f1[nans] = 0
#    cisf2[nans] = 0
#    cispadf2[nans] = 0
#    cispad_f2[nans] = 0
    for k in range(cohtimes):

        start = k * cohT[n]
        end = min((k + 1) * cohT[n], N)
#        print(np.sum(cisf1[start:end] * signal[4, start:end]) / np.sum(signal[4, start:end]))
#        print(np.sum(cisf1[start:end] * signal[5, start:end]) / np.sum(signal[5, start:end]))
#        print(np.sum(cisf1[start:end] * signal[6, start:end]) / np.sum(signal[6, start:end]))
#        print(np.sum(cispadf1[start:end] * signal[4, start:end]) / np.sum((cospad[start:end] + 1j * sinpad[start:end]) * signal[4, start:end]))
#        print(np.sum(cispadf1[start:end] * signal[5, start:end]) / np.sum((cospad[start:end] + 1j * sinpad[start:end]) * signal[5, start:end]))
#        print(np.sum(cispadf1[start:end] * signal[6, start:end]) / np.sum((cospad[start:end] + 1j * sinpad[start:end]) * signal[6, start:end]))
#        print(np.sum(cispad_f1[start:end] * signal[4, start:end]) / np.sum((cospad[start:end] + 1j * sinpad[start:end]) * signal[4, start:end]))
#        print(np.sum(cispad_f1[start:end] * signal[5, start:end]) / np.sum((cospad[start:end] + 1j * sinpad[start:end]) * signal[5, start:end]))
#        print(np.sum(cispad_f1[start:end] * signal[6, start:end]) / np.sum((cospad[start:end] + 1j * sinpad[start:end]) * signal[6, start:end]))
#        print(np.sum(cisf2[start:end] * signal[4, start:end]) / np.sum(signal[4, start:end]))
#        print(np.sum(cisf2[start:end] * signal[5, start:end]) / np.sum(signal[5, start:end]))
#        print(np.sum(cisf2[start:end] * signal[6, start:end]) / np.sum(signal[6, start:end]))
#        print(np.sum(cispadf2[start:end] * signal[4, start:end]) / np.sum((cospad[start:end] + 1j * sinpad[start:end]) * signal[4, start:end]))
#        print(np.sum(cispadf2[start:end] * signal[5, start:end]) / np.sum((cospad[start:end] + 1j * sinpad[start:end]) * signal[5, start:end]))
#        print(np.sum(cispadf2[start:end] * signal[6, start:end]) / np.sum((cospad[start:end] + 1j * sinpad[start:end]) * signal[6, start:end]))
#        print(np.sum(cispad_f2[start:end] * signal[4, start:end]) / np.sum((cospad[start:end] + 1j * sinpad[start:end]) * signal[4, start:end]))
#        print(np.sum(cispad_f2[start:end] * signal[5, start:end]) / np.sum((cospad[start:end] + 1j * sinpad[start:end]) * signal[5, start:end]))
#        print(np.sum(cispad_f2[start:end] * signal[6, start:end]) / np.sum((cospad[start:end] + 1j * sinpad[start:end]) * signal[6, start:end]))

        muxfd[n][k, 0] = math.pi * R * math.sqrt(2 * rho) / 4 * np.sum(cis[start:end] * (1 - signal[0, start:end] + 1j * signal[1, start:end]))
        muxfd[n][k, 1] = math.pi * R * math.sqrt(2 * rho) / 4 * np.sum(cis[start:end] * (signal[1, start:end] + 1j * signal[0, start:end]))
        muxfd[n][k, 2] = math.pi * R * math.sqrt(2 * rho) / 4 * np.sum(cis[start:end] * (signal[3, start:end] - 1j * signal[4, start:end]))
        muxfd[n][k, 3] = math.pi * R * math.sqrt(2 * rho) / 4 * np.sum(cis[start:end] * (-signal[4, start:end] + 1j * (signal[2, start:end] - signal[3, start:end])))
        muxfd[n][k, 4] = math.pi * R * math.sqrt(2 * rho) / 4 * np.sum(cis[start:end] * (signal[5, start:end] - 1j * signal[6, start:end]))

        mux0[n][k, 0] = math.pi * R * math.sqrt(rho / 2) * (np.sum(cosfd[start:end]) - np.sum(cosfd[start:end] * signal[0, start:end]) + np.sum(sinfd[start:end] * signal[1, start:end]))
        mux0[n][k, 1] = math.pi * R * math.sqrt(rho / 2) * (np.sum(cosfd[start:end] * signal[1, start:end]) + np.sum(sinfd[start:end] * signal[0, start:end]))
        mux0[n][k, 2] = math.pi * R * math.sqrt(rho / 2) * (np.sum(cosfd[start:end] * signal[3, start:end]) - np.sum(sinfd[start:end] * signal[4, start:end]))
        mux0[n][k, 3] = math.pi * R * math.sqrt(rho / 2) * (-np.sum(cosfd[start:end] * signal[4, start:end]) + np.sum(sinfd[start:end] * (signal[2, start:end] - signal[3, start:end])))
        mux0[n][k, 4] = math.pi * R * math.sqrt(rho / 2) * (np.sum(cosfd[start:end] * signal[5, start:end]) - np.sum(sinfd[start:end] * signal[6, start:end]))

        muyfd[n][k, 0] = math.pi * R * math.sqrt(2 * rho) / 4 * np.sum(cis[start:end] * (signal[1, start:end] - 1j * (1 - signal[0, start:end])))
        muyfd[n][k, 1] = math.pi * R * math.sqrt(2 * rho) / 4 * np.sum(cis[start:end] * (signal[0, start:end] - 1j * signal[1, start:end]))
        muyfd[n][k, 2] = math.pi * R * math.sqrt(2 * rho) / 4 * np.sum(cis[start:end] * (-signal[4, start:end] - 1j * signal[3, start:end]))
        muyfd[n][k, 3] = math.pi * R * math.sqrt(2 * rho) / 4 * np.sum(cis[start:end] * (signal[2, start:end] - signal[3, start:end] + 1j * signal[4, start:end]))
        muyfd[n][k, 4] = math.pi * R * math.sqrt(2 * rho) / 4 * np.sum(cis[start:end] * (-signal[6, start:end] - 1j * signal[5, start:end]))

        muy0[n][k, 0] = math.pi * R * math.sqrt(rho / 2) * (np.sum(cosfd[start:end] * signal[1, start:end]) - np.sum(sinfd[start:end]) + np.sum(sinfd[start:end] * signal[0, start:end]))
        muy0[n][k, 1] = math.pi * R * math.sqrt(rho / 2) * (np.sum(cosfd[start:end] * signal[0, start:end]) - np.sum(sinfd[start:end] * signal[1, start:end]))
        muy0[n][k, 2] = math.pi * R * math.sqrt(rho / 2) * (-np.sum(cosfd[start:end] * signal[4, start:end]) - np.sum(sinfd[start:end] * signal[3, start:end]))
        muy0[n][k, 3] = math.pi * R * math.sqrt(rho / 2) * (np.sum(cosfd[start:end] * (signal[2, start:end] - signal[3, start:end])) + np.sum(sinfd[start:end] * signal[4, start:end]))
        muy0[n][k, 4] = math.pi * R * math.sqrt(rho / 2) * (-np.sum(cosfd[start:end] * signal[6, start:end]) - np.sum(sinfd[start:end] * signal[5, start:end]))

        muzfd[n][k, 2] = math.pi * R * math.sqrt(rho / 2) * np.sum((cospad[start:end] + 1j * sinpad[start:end]) * signal[5, start:end])
        muzfd[n][k, 3] = -math.pi * R * math.sqrt(rho / 2) * np.sum((cospad[start:end] + 1j * sinpad[start:end]) * signal[6, start:end])
        muzfd[n][k, 4] = math.pi * R * math.sqrt(rho / 2) * np.sum((cospad[start:end] + 1j * sinpad[start:end]) * (1 - signal[2, start:end]))

        muz0[n][k, 2] = math.pi * R * math.sqrt(rho / 2) * np.sum(signal[5, start:end])
        muz0[n][k, 3] = -math.pi * R * math.sqrt(rho / 2) * np.sum(signal[6, start:end])
        muz0[n][k, 4] = math.pi * R * math.sqrt(rho / 2) * (end - start - np.count_nonzero(np.logical_and(nans >= start, nans < end)) - np.sum(signal[2, start:end]))
del signal
print(str(time.time() - t0) + ' ' + str(process.memory_info().rss))

def compute_power(year):
    start = (365 * year + math.floor((year + (skipyears + 1) % 4) / 4)) * 24 * 60
    end = (365 * (year + 1) + math.floor((year + 1 + (skipyears + 1) % 4) / 4)) * 24 * 60
    subseries = timeseries[:,start:end]
    power = np.zeros((5, 5, 2 * uptime), dtype = complex)
    num_chunks = math.floor((end - start + downtime) / (uptime + downtime))
    chunksize = math.floor((end - start + downtime) / num_chunks) - downtime
    chunkmod = (end - start + downtime) % num_chunks
    chunks = [[k * (chunksize + downtime) + min(k, chunkmod), k * (chunksize + downtime) + min(k + 1, chunkmod) + chunksize] for k in range(num_chunks)]
    for chunk in chunks:
        sample = np.append(subseries[:,chunk[0]:chunk[1]], np.zeros((5, 2 * uptime + chunk[0] - chunk[1])), axis = 1)
        nancount = np.count_nonzero(np.logical_and(nans >= chunk[0] + start, nans < chunk[1] + start))
        if nancount != chunk[1] - chunk[0]:
            fft = np.fft.fft(sample)
            power += 2 * fft[:,None] * np.conj(fft[None]) / (60 * (chunk[1] - chunk[0] - nancount))
        else:
            num_chunks -= 1
    power = power / num_chunks
    return(power)

if __name__ == '__main__':
    pool1 = mp.Pool(years)
    results1 = pool1.map_async(compute_power, range(years)).get()
    pool1.close()
    pool1.join()
print(str(time.time() - t0) + ' ' + str(process.memory_info().rss))

#powerfile = open('power_netcdf.csv', 'w', newline = '')
#powerwriter = csv.writer(powerfile, delimiter = ',', quotechar = '|')
#for year in range(years):
#    powerwriter.writerow(results1[year].flatten())
#powerfile.close()
#print(str(time.time() - t0) + ' ' + str(process.memory_info().rss))
#asdfasdfasdf

#powerfile = open('power.csv', newline = '')
#powerreader = csv.reader(powerfile, delimiter = ',', quotechar = '|')
#results1 = []
#for year in range(years):
#    results1.append(np.array([complex(x) for x in next(powerreader)]).reshape(5, 5, 2 * uptime))
#powerfile.close()
#print(str(time.time() - t0) + ' ' + str(process.memory_info().rss))

def data(n):
    cohtimes = math.ceil(N / cohT[n])
    partitions = np.zeros((cohtimes, years))
    count = 0
    cohk = 0
    for year in range(years):
        yearlength = 365 * 24 * 60
        if (year + skipyears - 2) % 4 == 0: yearlength += 24 * 60
        if count + yearlength <= cohT[n]:
            partitions[cohk, year] += yearlength
            count += yearlength
        else:
            partitions[cohk, year] += cohT[n] - count
            partitions[cohk + 1, year] += yearlength - cohT[n] + count
            cohk += 1
            count += yearlength - cohT[n]
    for i in range(len(nans)):
        partitions[np.floor(nans[i] / cohT[n]).astype(int), nanyears[i]] -= 1

    ffts = np.zeros((cohtimes, 5, freqbounds[n][1] - freqbounds[n][0] + 2 * pad(n)), dtype = complex)
    for k in range(cohtimes):
        if k == cohtimes - 1: subseries = np.concatenate([timeseries[:, (cohtimes - 1) * cohT[n]:], np.zeros((5, cohtimes * cohT[n] - N))], axis = 1)
        else: subseries = timeseries[:, k * cohT[n] : (k + 1) * cohT[n]]
        fft = np.fft.fft(subseries)
        if n == 0: ffts[k] = np.concatenate([fft[:,-pad(n):], fft[:,:freqbounds[n][1] + pad(n)]], axis = 1)
        elif n == cohTs - 1: ffts[k] = np.concatenate([fft[:, freqbounds[n][0] - pad(n):], fft[:,:pad(n)]], axis = 1)
        else: ffts[k] = fft[:, freqbounds[n][0] - pad(n) : freqbounds[n][1] + pad(n)]
#    print(str(n) + ' ' + str(cohT[n]) + ' ' + str(time.time() - t0) + ' ' + str(process.memory_info().rss))
#    sys.stdout.flush()
    return(ffts, partitions)

if __name__ == '__main__':
    pool2 = mp.Pool(cohTs)
    results2 = pool2.map_async(data, range(cohTs)).get()
    pool2.close()
    pool2.join()
del timeseries
print(str(time.time() - t0) + ' ' + str(process.memory_info().rss))

X = []
for n in range(cohTs):
    X.append(np.moveaxis(results2[n][0], 2, 0))
powinterp = []
for year in range(years):
    powinterp.append(interp.interp1d(powerfreqs, np.append(results1[year], results1[year][:,:,0,None], axis = 2), axis = 2))
Sigma = []
for n in range(cohTs):
    if n == 0: padded_freqs = np.concatenate([np.arange(cohT[n] - pad(n), cohT[n]), np.arange(freqbounds[n][1] + pad(n))]) / (60 * cohT[n])
    elif n == cohTs - 1: padded_freqs = np.concatenate([np.arange(freqbounds[n][0] - pad(n), cohT[n]), np.arange(pad(n))]) / (60 * cohT[n])
    else: padded_freqs = np.arange(freqbounds[n][0] - pad(n), freqbounds[n][1] + pad(n)) / (60 * cohT[n])
    Sigma.append(np.moveaxis(np.tensordot(results2[n][1], [powinterp[year](padded_freqs) for year in range(years)], axes = 1) * 60 / 2, 3, 0))
del results1, results2, powinterp
print(str(time.time() - t0) + ' ' + str(process.memory_info().rss))

invA = [np.linalg.inv(np.linalg.cholesky(Sigma[n])) for n in range(cohTs)]
del Sigma
Y = [np.matmul(invA[n], X[n][...,None])[...,0] for n in range(cohTs)]
del X
nuxplus = [freqs[n][:,None,None] * np.matmul(invA[n][:-2 * pad(n)], muxfd[n][None,...,None])[...,0] for n in range(cohTs)]
nux0 = [freqs[n][:,None,None] * np.matmul(invA[n][pad(n) : -pad(n)], mux0[n][None,...,None])[...,0] for n in range(cohTs)]
nuxminus = [freqs[n][:,None,None] * np.matmul(invA[n][2 * pad(n):], np.conj(muxfd[n][None,...,None]))[...,0] for n in range(cohTs)]
nuyplus = [freqs[n][:,None,None] * np.matmul(invA[n][:-2 * pad(n)], muyfd[n][None,...,None])[...,0] for n in range(cohTs)]
nuy0 = [freqs[n][:,None,None] * np.matmul(invA[n][pad(n) : -pad(n)], muy0[n][None,...,None])[...,0] for n in range(cohTs)]
nuyminus = [freqs[n][:,None,None] * np.matmul(invA[n][2 * pad(n):], np.conj(muyfd[n][None,...,None]))[...,0] for n in range(cohTs)]
nuzplus = [freqs[n][:,None,None] * np.matmul(invA[n][:-2 * pad(n)], muzfd[n][None,...,None])[...,0] for n in range(cohTs)]
nuz0 = [freqs[n][:,None,None] * np.matmul(invA[n][pad(n) : -pad(n)], muz0[n][None,...,None])[...,0] for n in range(cohTs)]
nuzminus = [freqs[n][:,None,None] * np.matmul(invA[n][2 * pad(n):], np.conj(muzfd[n][None,...,None]))[...,0] for n in range(cohTs)]
del invA, muxfd, mux0, muyfd, muy0, muzfd, muz0

Nu = [np.stack([np.block([nuxplus[n], nux0[n], nuxminus[n]]), np.block([nuyplus[n], nuy0[n], nuyminus[n]]), np.block([nuzplus[n], nuz0[n], nuzminus[n]])], axis = 3) for n in range(cohTs)]
del nuxplus, nux0, nuxminus, nuyplus, nuy0, nuyminus, nuzplus, nuz0, nuzminus
U = []
S = []
for n in range(cohTs):
    u, s, vh = np.linalg.svd(Nu[n], full_matrices = False)
    U.append(u)
    S.append(s)
del Nu
Z = [np.matmul(np.conj(np.swapaxes(U[n], 2, 3)), np.block([Y[n][:-2 * pad(n)], Y[n][pad(n) : -pad(n)], Y[n][2 * pad(n):]])[...,None])[...,0] for n in range(cohTs)]
del Y, U
print(str(time.time() - t0) + ' ' + str(process.memory_info().rss))

#print(np.linalg.norm(Z[41][35488] / S[41][35488], axis = 1))
#print(np.linalg.norm(Z[45][98] / S[45][98], axis = 1))

outfile = open(filename, 'w', newline = '')
writer = csv.writer(outfile, delimiter = ',', quotechar = '|')
writer.writerow([int(math.ceil(N / cohT[n])) for n in range(cohTs)])
for n in range(cohTs):
    writer.writerow(freqs[n])
    for f in range(len(freqs[n])):
        writer.writerow(S[n][f].flatten())
        writer.writerow(Z[n][f].flatten())
outfile.close()
print(str(time.time() - t0) + ' ' + str(process.memory_info().rss))
