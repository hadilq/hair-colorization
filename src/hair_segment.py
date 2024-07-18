from typing import List, Dict, Tuple
import numpy as np
import os
from ultralytics import YOLO
import cv2 as cv

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(dir_path + '/../model/hair_segment.pt')

class HairSegment:
    def load_model(self, model_path=model_path):
        self.model = YOLO(model_path)

    def find_mask(self, original_image_path):
        img_rgb, masks = self.find_contours(original_image_path)
        if img_rgb is None:
            return img_rgb, None, None, None
        b_mask = np.zeros(img_rgb.shape[:2], np.uint8)
        if masks is None:
            return img_rgb, b_mask, None, None

        self.fill_b_mask(b_mask, masks)
        area = self.calculate_area(b_mask)
        hairy_label = set()
        while area and not hairy_label:
            expanded_b_mask = self.fill_expanded_b_mask(area, b_mask)

            vectorized_pixels, map_xy_to_position = self.prepare_vectorized_pixels(img_rgb, expanded_b_mask)
            log(3, "vectorized_pixels len: {0}", len(vectorized_pixels))
            label, K = self.optimize_k_means(vectorized_pixels)

            label_img = self.create_label_img(img_rgb.shape[:2], label, map_xy_to_position)
            log(3, "label_img shape: {0}", label_img.shape)

            hairy_label = self.find_hairy_label(b_mask, K, label_img)
            log(3, "hairy_label: {0}", hairy_label)
            area  *= 2/3

        self.remove_not_hairy_from_b_mask(b_mask, label_img, hairy_label)

        return img_rgb, b_mask, label_img, hairy_label

    def find_contours(self, original_image_path):
        # Run batched inference on a list of images
        original_image = cv.imread(original_image_path)
        if original_image is None or original_image.size == 0:
            return original_image, None
        result = self.model(original_image).pop()  # return a list of Results objects

        masks = result.masks

        img_rgb = cv.cvtColor(original_image, cv.COLOR_BGR2RGB)
        return (img_rgb, masks)

    def fill_b_mask(self, b_mask, masks):
        for contour in masks.xy:
            contour = contour.astype(np.int32)
            contour = contour.reshape(-1, 1, 2) # make it into a shape that drawContours prefers
            _ = cv.drawContours(b_mask, [contour], -1, (255, 255, 255), cv.FILLED)

    def calculate_area(self, b_mask):
        area = 0
        for y, b_mask_y in enumerate(b_mask):
            for x, mask in enumerate(b_mask_y):
                if mask:
                    area += 1
        return area

    def fill_expanded_b_mask(self, area, b_mask):
        ## apply convolution to expand the mask area to give room for non-hair colors to dominate
        expand_factor = int(np.ceil((np.sqrt(2) - 1.0) * np.sqrt(area)))
        # have a circle kernel
        log(3, "area: {0}", area)
        log(3, "expand_factor: {0}", expand_factor)
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(expand_factor, expand_factor))
        expanded_b_mask = cv.filter2D(b_mask, -1, kernel)

        return expanded_b_mask

    def prepare_vectorized_pixels(self, img, expanded_b_mask):
        map_xy_to_position = {}
        vectorized_pixels = []
        for y, mask_y in enumerate(expanded_b_mask):
            for x, mask in enumerate(mask_y):
                if mask:
                    map_xy_to_position[(x, y)] = len(vectorized_pixels)
                    vectorized_pixels.append(img[y][x])

        vectorized_pixels = np.array(vectorized_pixels)
        vectorized_pixels = np.float32(vectorized_pixels)

        return (vectorized_pixels, map_xy_to_position)

    def optimize_k_means(self, vectorized):
        if len(vectorized) < 2:
            log(3, "vectorized's len is {0}", vectorized)
            return np.array([0] * len(vectorized)), len(vectorized)

        def silhouette_coefficient(label, center):
            a, b = 0, 0
            for i, p in enumerate(vectorized):
                a += cv.norm(p - center[label[i][0]])
                nearest = float("inf")
                for j, c in enumerate(center):
                    if j != label[i][0]:
                        nearest = min(nearest, cv.norm(p - c))
                b += nearest
            return (b - a) / max(a, b)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        attempts = 10

        prev_silhouette_score = float("inf")
        for K in range(3, min(500, len(vectorized))):
            log(3, "K: {0}", K)
            compactness, label, center = cv.kmeans(vectorized, K, None, criteria, attempts, cv.KMEANS_PP_CENTERS)
            silhouette_score = silhouette_coefficient(label, center)
            if silhouette_score > prev_silhouette_score:
                # reached optimal K
                return label, K

            prev_silhouette_score = silhouette_score

        return label, K

    def create_label_img(self, img_shape, label, map_xy_to_position):
        # it'll be a matrix full of 255
        label_img = np.full(img_shape, -1, np.uint8)
        for y in range(label_img.shape[0]):
            for x in range(label_img.shape[1]):
                if (x, y) in map_xy_to_position:
                    position = map_xy_to_position[(x, y)]
                    label_img[y][x] = label[position][0]
        return label_img

    def find_hairy_label(self, b_mask, K, label_img):
        count_hair_label = [0] * K
        count_not_hair_label = [0] * K
        for y, l_y in enumerate(label_img):
            for x, l in enumerate(l_y):
                if l != 255:
                    if b_mask[y][x]:
                        count_hair_label[l] += 1
                    else:
                        count_not_hair_label[l] += 1

        log(3, "count_hair_label: {0}", count_hair_label)
        log(3, "count_not_hair_label: {0}", count_not_hair_label)
        hairy_label = set()

        for i in range(K):
            if count_hair_label[i] > count_not_hair_label[i]:
                hairy_label.add(i)
        return hairy_label

    def remove_not_hairy_from_b_mask(self, b_mask, label_img, hairy_label):
        """
            Modify b_mask to only contains hairy pixels
        """
        for y, l_y in enumerate(label_img):
            for x, l in enumerate(l_y):
                if l != 255 and l not in hairy_label and b_mask[y][x]:
                    b_mask[y][x] = 0

    def confine_label_img_to_hairy_ones(self, b_mask, label_img, hairy_label):
        """
            Modify label_img to only contains hairy labels
        """
        for y, m_y in enumerate(b_mask):
            for x, m in enumerate(m_y):
                if label_img[y][x] != 255 or label_img[y][x] not in hairy_label or not m:
                    label_img[y][x] = 255

    def label_sampling(self, img, label_img, hairy_label):
        """
            Modify label_img to put sample data instead of meaningless label number.
        """
        img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        label_sample_img = np.full((img.shape[0], img.shape[1], 6), -1, dtype = np.uint8)
        for l in hairy_label:
            sample_saturation = []
            sample_value = []
            for y, l_y in enumerate(label_img):
                for x, pixel_lable in enumerate(l_y):
                    if pixel_lable == l:
                        sample_saturation.append(img_hsv[y][x][1])
                        sample_value.append(img_hsv[y][x][2])
            if not sample_saturation or not sample_value:
                continue
            data = self.calculate_data(sample_saturation)
            data.extend(self.calculate_data(sample_value))
            data = np.array(data, dtype = np.uint8)

            for y, l_y in enumerate(label_img):
                for x, pixel_lable in enumerate(l_y):
                    if pixel_lable == l:
                        label_sample_img[y][x] = data
        return label_sample_img

    def calculate_data(self, sample):
        """
            return a list of mean, median, and mod of the sample
        """
        if sample is None or len(sample) == 0:
            return [0, 0, 0]
        sample = np.sort(sample, axis=None)
        median = 0.0
        if len(sample) % 2 == 0:
            middle = len(sample) // 2
            median = (1.0 * sample[middle] + sample[middle + 1]) / 2
        else:
            median = 1.0 * sample[len(sample) // 2 + 1]

        mode, last_value = -1, -1
        count = 0
        max_count = 0
        mean = 0.0
        for value in sample:
            if value == last_value:
                count += 1
            else:
                count = 0
                last_value = value

            if count >= max_count:
                max_count = count
                mode = value

            mean += value

        mean = mean // len(sample)

        return [int(mean), int(median), int(mode)]

    def make_data(self, image_path):
        """
            Pre-process image and extract data.
            Return img, b_mask, data, label_img.
        """
        img, b_mask, label_img, hairy_label = self.find_mask(image_path)
        if label_img is None:
            return img, b_mask, None, None
        self.confine_label_img_to_hairy_ones(b_mask, label_img, hairy_label)
        label_img = self.label_sampling(img, label_img, hairy_label)
        if img is None or img.size == 0:
            log(3, "cannot find mask!")
            return img, b_mask, None, None

        img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        sample_in_hue = []
        for y in range(b_mask.shape[0]):
          for x in range(b_mask.shape[1]):
            if b_mask[y][x]:
              sample_in_hue.append(img_hsv[x][y])

        if len(sample_in_hue) == 0:
            return img, b_mask, None, None

        sample_in_hue = cv.cvtColor(np.array([sample_in_hue]), cv.COLOR_RGB2HSV)[0][:, :1]
        mean_hue, median_hue, mode_hue = self.calculate_data(sample_in_hue)

        masked_area = 0
        for y, m_y in enumerate(b_mask):
            for x, m in enumerate(m_y):
                if m:
                    masked_area += 1

        data = {
            'mean_hue': mean_hue,
            'median_hue': median_hue,
            'mode_hue': mode_hue,
            'masked_area': masked_area,
        }
        log(3, "data: {0}", data)
        return img, b_mask, data, label_img

    def make_gray_hair(self, img, b_mask):
        gray_img = img.copy()
        gray_img = cv.cvtColor(gray_img, cv.COLOR_RGB2GRAY)
        return gray_img

    def test_if_all_files_are_parcelable(self, input_dir):
        import glob
        import json
        for json_path in glob.glob(input_dir+ f'/*.json'):
            with open(json_path, 'r') as f:
                try:
                    data = json.loads(f.read())
                    log(1, "All good: {0}", json_path)
                except json.JSONDecodeError as e:
                    log(2, "Invalid JSON syntax: {0}", e)
                    log(2, "json path: {0}", json_path)
                    os.remove(json_path)
        for png_path in glob.glob(input_dir+ f'/*.png'):
            b_mask = cv.imread(png_path)
            if b_mask is None:
                log(2, "png path: {0}", png_path)
                os.remove(png_path)



## Helper function to predict
def predict(image_path):
    hair_segment_predictor = HairSegmentPredictor()
    hair_segment_predictor.setup()

    img, b_mask, _, _ = hair_segment_predictor.find_mask(image_path)
    if img is None or img.size == 0:
        return None

    for y in range(b_mask.shape[0]):
      for x in range(b_mask.shape[1]):
        if b_mask[y][x]:
          img[y][x] = [255,255,255]
    return img


## Helper function to predict and show
def predict_and_show(image_path):
    img = predict(image_path)
    if img is None:
        log(4, "prediction is empty! {0}", image_path)
        return

    from PIL import Image as Img
    Img.fromarray(img).show()

## Helper function to predict and save
def predict_and_save(image_path, output):
    img = predict(image_path)
    if img is None:
        log(4, "prediction is empty! {0}", image_path)
        return

    from PIL import Image as Img
    Img.fromarray(img).save(output)

## Helper function to predict and save
def predicts_and_save(image_path_list, output):
    for image_path in image_path_list:
        image_name = os.path.basename(image_path)
        predict_and_save(image_path, output + '/' + image_name)

def log(level, s, *arg):
    if level > 2:
       if arg:
           print(s.format(*arg))
       else:
           print(s)

