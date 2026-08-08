import cv2
import numpy as np



def resizeForCnn(image, size):
	TARGET_IMAGE_SIZE = size
	HEIGHT = image.shape[0]
	WIDTH = image.shape[1]

	# Calculate multiplier that makes longest side = TARGET_IMAGE_SIZE
	# og*0.5 = 244
	# 0.5 = 244/og
	scale = TARGET_IMAGE_SIZE/max(HEIGHT, WIDTH)

	NEW_HEIGHT = int(HEIGHT * scale)
	NEW_WIDTH = int(WIDTH * scale)

	halfScaled = cv2.resize(image, (NEW_WIDTH, NEW_HEIGHT))


	# center image on black canvas of size 244x244 
	zerosTemplate = np.zeros((244, 244, 3), dtype=np.uint8)

	topOffset = (TARGET_IMAGE_SIZE - NEW_HEIGHT) // 2
	leftOffset = (TARGET_IMAGE_SIZE - NEW_WIDTH) // 2

	zerosTemplate[topOffset:(topOffset + NEW_HEIGHT), leftOffset:(leftOffset+NEW_WIDTH)] = halfScaled 
	print(halfScaled.dtype)

	resized = zerosTemplate

	return resized


# image path
path = "C:\addPath"
#path = "/home/jp/college/CPECOG1_ComputerVision/project/test.png"

image = cv2.imread(path)
cv2.imshow("Input", image)

resized = resizeForCnn(image, 244)

cv2.imshow("Test Preprocess", resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
