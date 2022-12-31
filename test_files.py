import numpy as np

# Create a NumPy array
array = np.array([[1, 2, 3], [4, 5, 6]])

# Write the array to a CSV file
np.savetxt("array.csv", array, delimiter=",", comments="# Hi all...")

# Read the CSV file into an array
array = np.loadtxt("array.csv", delimiter=",")

print(array)
