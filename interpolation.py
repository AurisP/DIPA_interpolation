# Import packages
import cv2
import skimage.io

# Import some functions to the global namespace for convenience
from scipy.ndimage import *
from scipy.fft import *
from scipy.signal import * 
from scipy.stats import *
from skimage.filters import *
import numpy as np


def nearest_neighbour_interpolation(img, scale_percent):
    # Calculate the new dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # Create an empty array to hold the resized image
    resized_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the scaling factor for each dimension
    scale_x = img.shape[1] / resized_img.shape[1]
    scale_y = img.shape[0] / resized_img.shape[0]

    # Perform nearest neighbour interpolation
    for y in range(resized_img.shape[0]):
        for x in range(resized_img.shape[1]):
            src_x = int(x * scale_x)
            src_y = int(y * scale_y)
            resized_img[y, x] = img[src_y, src_x]

    return resized_img


def bilinear_interpolation(img, scale_percent):
    # Calculate the new dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # Create an empty array to hold the resized image
    resized_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the scaling factor for each dimension
    scale_x = img.shape[1] / resized_img.shape[1]
    scale_y = img.shape[0] / resized_img.shape[0]

    # Perform bilinear interpolation
    for y in range(height):
        for x in range(width):
             # Calculate the coordinates in the original image
            src_x = x * scale_x
            src_y = y * scale_y

            # Calculate the integer coordinates of the 4 nearest pixels
            x1 = int(np.floor(src_x))
            y1 = int(np.floor(src_y))
            x2 = min(x1 + 1, img.shape[1] - 1)
            y2 = min(y1 + 1, img.shape[0] - 1)

            # Calculate the fractional distances to the 4 nearest pixels
            tx = src_x - x1
            ty = src_y - y1

            # Calculate the pixel values of the 4 nearest pixels
            pixel1 = img[max(y1, 0), max(x1, 0)]
            pixel2 = img[max(y1, 0), min(x2, img.shape[1] - 1)]
            pixel3 = img[min(y2, img.shape[0] - 1), max(x1, 0)]
            pixel4 = img[min(y2, img.shape[0] - 1), min(x2, img.shape[1] - 1)]

            # Interpolate the pixel value using the fractional distances
            interpolated_pixel = (1 - tx) * (1 - ty) * pixel1 \
                + tx * (1 - ty) * pixel2 \
                + (1 - tx) * ty * pixel3 \
                + tx * ty * pixel4

            # Assign the interpolated pixel value to the resized image
            resized_img[y, x] = interpolated_pixel.astype(np.uint8)

    return resized_img

def bicubic_interpolation(img, scale_percent):
    # Calculate the new dimensions
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)

    # Create an empty array to hold the resized image
    resized_img = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the scaling factor for each dimension
    scale_x = img.shape[1] / resized_img.shape[1]
    scale_y = img.shape[0] / resized_img.shape[0]

    # Perform bicubic interpolation
    for y in range(height):
        for x in range(width):
            # Calculate the coordinates in the original image
            src_x = x * scale_x
            src_y = y * scale_y

            # Calculate the integer coordinates of the 16 nearest pixels
            x1 = int(np.floor(src_x)) - 1
            y1 = int(np.floor(src_y)) - 1
            x2 = min(x1 + 1, img.shape[1] - 1)
            y2 = min(y1 + 1, img.shape[0] - 1)
            x3 = min(x1 + 2, img.shape[1] - 1)
            y3 = min(y1 + 2, img.shape[0] - 1)
            x4 = min(x1 + 3, img.shape[1] - 1)
            y4 = min(y1 + 3, img.shape[0] - 1)

            # Calculate the fractional distances to the nearest pixels
            tx = src_x - np.floor(src_x)
            ty = src_y - np.floor(src_y)

            # Calculate the bicubic interpolation coefficients for each row
            coeff_row1 = bicubic_coeffs(img, y1, x1)
            coeff_row2 = bicubic_coeffs(img, y2, x1)
            coeff_row3 = bicubic_coeffs(img, y3, x1)
            coeff_row4 = bicubic_coeffs(img, y4, x1)

            # Interpolate the pixel value using the bicubic interpolation coefficients
            interpolated_pixel = (bicubic_interpolate(coeff_row1, tx) +
                                  bicubic_interpolate(coeff_row2, tx) +
                                  bicubic_interpolate(coeff_row3, tx) +
                                  bicubic_interpolate(coeff_row4, tx)) / 4.0

            # Assign the interpolated pixel value to the resized image
            resized_img[y, x] = np.clip(interpolated_pixel, 0, 255).astype(np.uint8)

    return resized_img

def bicubic_coeffs(img, row, col):
    # Compute the bicubic interpolation coefficients for a row or column of the image
    v = np.array([-1, 0, 1, 2])
    x = col + v
    y = row + v
    coeffs = np.zeros((4, 4))

    for i in range(4):
        for j in range(4):
            xj = max(min(x[j], img.shape[1] - 1), 0)
            yi = max(min(y[i], img.shape[0] - 1), 0)
            coeffs[i, j] = img[yi, xj]

    return bicubic_solve(coeffs)


def bicubic_solve(coeffs):
    # Solve for the bicubic interpolation coefficients using linear regression
    A = np.array([[1, 0, 0, 0],
                  [1, 1, 1, 1],
                  [1, 2, 4, 8],
                  [1, 3, 9, 27]])
    B = np.array([coeffs[0, :], coeffs[1, :], coeffs[2, :], coeffs[3, :]])
    C = np.linalg.inv(A).dot(B)

    return C

# Load the image
# img = cv2.imread('medalja_dubrovnik.png')
img = cv2.imread('uzorak.tif')
# Define the scaling factor
scale_percent = 200  # increase the size by 200%

# Call the function to perform nearest neighbour interpolation
# resized_img = nearest_neighbour_interpolation(img, scale_percent)
resized_img = bilinear_interpolation(img, scale_percent)
# resized_img = bicubic_interpolation(img, scale_percent)
# Display the original and resized image
cv2.imshow('Original Image', img)
cv2.imshow('Resized Image', resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()