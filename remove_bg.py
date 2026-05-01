import cv2
import numpy as np

img = cv2.imread('public/assets/approach-step-5.jpg')
h, w = img.shape[:2]

img_bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
mask = np.zeros((h+2, w+2), np.uint8)

# 25, 25, 25 tolerance
lo = (15, 15, 15)
hi = (15, 15, 15)

# flags: 4 connectivity, 1 fill value, cv2.FLOODFILL_FIXED_RANGE (1<<16)
flags = 4 | (1 << 8) | (1 << 16)

cv2.floodFill(img, mask, (0, 0), (0, 0, 0), lo, hi, flags)
cv2.floodFill(img, mask, (w-1, 0), (0, 0, 0), lo, hi, flags)
cv2.floodFill(img, mask, (0, h-1), (0, 0, 0), lo, hi, flags)
cv2.floodFill(img, mask, (w-1, h-1), (0, 0, 0), lo, hi, flags)

bg_mask = mask[1:-1, 1:-1]

# Dilate the background mask by a couple of pixels to eat the black fringe
kernel = np.ones((3,3), np.uint8)
bg_mask = cv2.dilate(bg_mask, kernel, iterations=2)

img_bgra[bg_mask == 1, 3] = 0

# Smooth the alpha channel for anti-aliased edges
alpha = img_bgra[:, :, 3]
alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
img_bgra[:, :, 3] = alpha

cv2.imwrite('public/assets/approach-step-5-transparent.png', img_bgra)
