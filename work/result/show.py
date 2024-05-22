import matplotlib.pyplot as plt

# Example data
x = [1, 2, 3, 4, 5]
y = [2, 4, 4, 5, 5]
y2 = [1, 4, 4, 6, 7]
# Creating the line plot with labels and legend
plt.figure(figsize=(8, 6))
plt.plot(x, y, marker='o', label='Line 1')  # Adding a label and a line
plt.plot(x, y2, marker='o', label='Line 2')
plt.title('Line Plot Example with Labels and Legend')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.legend()  # Adding the legend
plt.show()
