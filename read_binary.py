import numpy as np
import matplotlib.pyplot as plt

# An example script for reading binary data

# Open the file "CAN_0" for reading in binary mode "rb"
filename = "CAN_0"
file = open("CAN_0", "rb")

# Read in all bytes
bytes = file.read()

# Convert bytes to f32
data = np.frombuffer(bytes, dtype=np.float32)

# Separate into three chunks
chunks = np.split(data, 3)

# Plot each chunk
fig, axes = plt.subplots(3, 1)
axes[0].plot(chunks[0])
axes[1].plot(chunks[1])
axes[2].plot(chunks[2])
axes[2].set_xlabel("second of day")
plt.savefig(f"{filename}")

# Print several elements
ELEMENTS = 10
print(f"{chunks[0][:10]}") 
print(f"{chunks[1][:10]}") 
print(f"{chunks[2][:10]}") 