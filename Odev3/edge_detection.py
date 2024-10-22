import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import correlate


square = cv2.imread('images/square.png')
circle = cv2.imread('images/circle.png')

square = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
circle = cv2.cvtColor(circle, cv2.COLOR_BGR2GRAY)

horizontal_kernel = np.array([[-1, 1]])
vertical_kernel = horizontal_kernel.T

filter_horizontal_square = correlate(square, horizontal_kernel)
filter_vertical_square = correlate(square, vertical_kernel)
filter_square = np.sqrt(np.power(filter_horizontal_square, 2) + np.power(filter_vertical_square, 2))

plt.imshow(filter_square, cmap='gray')
plt.show()


filter_horizontal_circle = correlate(circle, horizontal_kernel)
filter_vertical_circle = correlate(circle, vertical_kernel)
filter_circle = np.sqrt(np.power(filter_horizontal_circle, 2) + np.power(filter_vertical_circle, 2))

plt.imshow(filter_circle, cmap='gray')
plt.show()
