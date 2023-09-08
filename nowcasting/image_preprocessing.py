import numpy as np
import cv2
from functools import partial


# Preprocessing of the images from the camera to use them in the trained model
class PreProcessImage():

    def __init__(self):
        self.pipeline = [
                            partial(self.std_scale_image), 
                            partial(self.convert_to_uint8),
                            partial(self.crop_image),
                            partial(self.resize_and_stack_channels)
                        ]

        # TODO Figure out how to best set it up later - probably allow for calling it without init values, specify the paths for each image in the function
        # In this way it will be better suited to use it in the loop


    def crop_image(image: np.array) -> np.array:
        # The initial image is of the size [480x640]. This function crops it to a square with black corners

        # Cropping and applying a mask to filter the circle in the centre
        mask = np.zeros((480, 640), dtype=np.uint8)
        cv2.circle(mask, (320, 225), 192, (255), -1)  # Center is (320, 225) and radius is 192
        cv2.circle(mask, (323, 229), 12, (0), -1)

        pts = np.array([(320, 249), (312, 355), (309, 406), (309, 407), (309, 431), (327, 432), (327, 409), (327, 407), (327, 357), (326, 249), (325, 244), (330, 239), (317, 239), (322, 244)], np.int32)
        # pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], (0))

        cropped_image = cv2.bitwise_and(image, image, mask=mask)
        
        return cropped_image[33:417, 128:512]


    def std_scale_image(image: np.array) -> np.array:

        MAX_LIMIT = 45_000
        MIN_LIMIT = 35_000

        # Min limit - only outliers (sun) is below this limit
        image[image < MIN_LIMIT] = MIN_LIMIT

        # Max limit - only outliers during the day are above this limit
        image[image > MAX_LIMIT] = MAX_LIMIT


        (p25, p75) = np.percentile(image.flatten(), (25, 75))
        std = (p75 - p25) #np.std(image)
        mean = np.mean (image)

        scaled_image = (image - mean) / std                         # std scaling
        scaled_image = scaled_image + np.abs(np.min(scaled_image))  # adding min value to go above 0
        
        return scaled_image / np.max(scaled_image)                  # divide by max to bring to 0-1 range


    def log_filter(image: np.array) -> np.array:

        c = 255 / np.log(1 + np.max(image))
        log_image = c * (np.log(image + 1.01))
        log_image = np.array(log_image, dtype=np.uint8)

        return log_image


    def gamma_correction(image: np.array, gamma: float) -> np.array:

        gamma_corrected = np.array((255*(image / 255)) ** gamma, dtype = 'uint8')

        return gamma_corrected

    def convert_to_uint8(image: np.array) -> np.array: 
        # Converts image from floating point (0-1) to uint8 (0-255) range

        return (image * 255).astype('uint8')


    def resize_and_stack_channels(image: np.array) -> np.array:
        # The network is trained to feed in images with dimentions (128, 128, 3)
        
        resized_image = cv2.resize(image, (128, 128))

        return np.stack((resized_image, resized_image, resized_image), axis = 2)


    def execute_pipeline(self, pipeline: list, image: np.array) -> np.array:
        # Executes the prepared image transformation pipeline
        
        for step in self.pipeline:
            image = step(image)

        return image

    def transform_image(self, image: np.array):
        
        new_image = self.execute_pipeline(image)

        return new_image


# if __name__ == '__main__':

#     pipeline = [
#         partial(std_scale_image()), 
#         partial(convert_to_uint8()),
#         partial(crop_image()),
#         partial(resize_and_stack_channels())
#     ]

#     pass