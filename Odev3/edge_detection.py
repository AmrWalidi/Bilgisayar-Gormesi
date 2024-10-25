import cv2
import numpy as np
import matplotlib.pyplot as plt

square = cv2.imread('images/square.png').astype(np.float32)
circle = cv2.imread('images/circle.png').astype(np.float32)

square = cv2.cvtColor(square, cv2.COLOR_BGR2GRAY)
circle = cv2.cvtColor(circle, cv2.COLOR_BGR2GRAY)

horizontal_kernel = np.array([[-1, 1]])
vertical_kernel = horizontal_kernel.T

filter_horizontal_square = cv2.filter2D(square, -1, horizontal_kernel)
filter_vertical_square = cv2.filter2D(square, -1, vertical_kernel)
filter_square = np.sqrt(np.power(filter_horizontal_square, 2) + np.power(filter_vertical_square, 2))

plt.imshow(filter_square, cmap='gray')
plt.show()

filter_horizontal_circle = cv2.filter2D(circle, -1, horizontal_kernel)
filter_vertical_circle = cv2.filter2D(circle, -1, vertical_kernel)
filter_circle = np.sqrt(np.power(filter_horizontal_circle, 2) + np.power(filter_vertical_circle, 2))

plt.imshow(filter_circle, cmap='gray')
plt.show()
