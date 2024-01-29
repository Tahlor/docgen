import numpy as np
import matplotlib.pyplot as plt

# Define the range for line thickness
line_thickness_range = (1, 5)  # Example range

# Generate a large number of samples to plot the distribution
samples = 10000
line_thicknesses = []

for _ in range(samples):
    exp_number = np.random.exponential(1)
    line_thickness = int(exp_number) % (line_thickness_range[1] - line_thickness_range[0]) + line_thickness_range[0]
    line_thicknesses.append(line_thickness)
    print(line_thickness)

# Plotting the histogram
plt.hist(line_thicknesses, bins=range(min(line_thicknesses), max(line_thicknesses) + 2, 1), align='left')
plt.xlabel('Line Thickness')
plt.ylabel('Frequency')
plt.title('Distribution of Line Thickness')
plt.show()

