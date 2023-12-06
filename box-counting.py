import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('1.tif')
img_array = np.array(img)
def box_count(img_array, box_size):
    height, width = img_array.shape

    box_count = 0

    for i in range(0, height, box_size):
        for j in range(0, width, box_size):
            if np.sum(img_array[i:i+box_size, j:j+box_size]) > 0:
                box_count += 1

    return box_count

box_sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024,2448]#3109 3408
for size in box_sizes:
    count = box_count(img_array, size)
    print(f'{count}')

box_counts = []

for size in box_sizes:
    count = box_count(img_array, size)
    box_counts.append(count)

log_box_sizes = np.log(box_sizes)
log_box_counts = np.log(box_counts)

plt.figure(figsize=(10, 6))
plt.plot(log_box_sizes, log_box_counts, 'bo-')
plt.xlabel('log r')
plt.ylabel('log N')
plt.title('Log-Log Plot')
plt.grid(True, which='both', ls='--', c='gray')

coefficients = np.polyfit(log_box_sizes, log_box_counts, 1)

slope = coefficients[0]
print(f'斜率D: {slope}')

plt.plot(log_box_sizes, np.polyval(coefficients, log_box_sizes), 'r--')

plt.show()