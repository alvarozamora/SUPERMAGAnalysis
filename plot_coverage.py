import numpy as np
import matplotlib.pyplot as plt

# Open days with data
filename = "days_with_data"
file = open(filename, "rb")

# Read in all bytes
bytes = file.read()

# Convert bytes to ints
data = np.frombuffer(bytes, dtype=np.int64)

# Reshape array
data = data.reshape(-1,2)

# Plot data
plt.plot(data[:,0], data[:,1],'.')
plt.grid()
plt.xlabel("day")
plt.ylabel("station count")
plt.savefig("coverage.png")


##############



# Open days with data
filename = "secs_with_data"
file = open(filename, "rb")

# Read in all bytes
bytes = file.read()

# Convert bytes to ints
data = np.frombuffer(bytes, dtype=np.int64)

# Reshape array
data = data.reshape(-1,2)

# Find zeros
zeros = data[:,1] <= 3

# sort cumsum of 
# sort = np.argsort(data[:,0])
# cs = np.cumsum(zeros[sort])

# Plot data
plt.figure(figsize=(12,8))
plt.plot(data[:,0], data[:,1],'.')
# plt.hist(data[:,0][zeros])
# plt.plot(1998 + np.arange(len(cs[::1000]))/365/24/60/60, cs[::1000])
plt.grid()
plt.xlabel("year")
# plt.xlim(0, len(zeros))
# plt.ylim(0.2, 1e9)
plt.xticks(np.cumsum(np.ones(23)*365*24*60*60), [str(year) for year in range(1998,2020+1)])
# plt.gca().set_yscale('log')
# plt.ylabel("unnormalized CDF of zeros")
plt.ylabel("stations reporting")
plt.savefig("sec_coverage.png")


