import tensorflow as tf
from keras.layers import Conv2D
from keras.layers import UpSampling2D
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import RepeatVector
from keras.layers import Reshape
from keras.utils import plot_model
from keras.utils import PyDataset
from keras.layers import Concatenate
from keras.applications.vgg16 import VGG16
import numpy as np
import glob
import os
from os import path
import math
import json
import cv2 as cv

class HairColorizationTraining:

    def __init__(self, img_h, img_w):
        self.img_h, self.img_w = img_h, img_w

    def define_model(self):
        inputs = Input(shape=(self.img_h, self.img_w, 3,))
        mask_inputs = Input(shape=(self.img_h, self.img_w, 1,))
        hue_inputs = Input(shape=(1,))

        # encoder
        encoder_layers = [
            Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            Conv2D(128, (3,3), activation='relu', padding='same', strides=2),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same', strides=2),
            Conv2D(512, (3,3), activation='relu', padding='same'),
            Conv2D(512, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same'),
        ]
        decoder_layers = [
            Conv2D(128, (3,3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(32, (3,3), activation='relu', padding='same'),
            Conv2D(16, (3,3), activation='relu', padding='same'),
            Conv2D(3, (3, 3), activation='tanh', padding='same'),
            UpSampling2D((2, 2)),
        ]

        encoder_output = Concatenate(axis = 3)([inputs, mask_inputs])
        for layer in encoder_layers:
            concat_shape = (np.uint32(encoder_output.shape[1]), np.uint32(encoder_output.shape[2]), np.uint32(hue_inputs.shape[-1]))

            image_feature = RepeatVector((concat_shape[0] * concat_shape[1]).item())(hue_inputs)
            image_feature = Reshape(concat_shape)(image_feature)
            fusion_output = Concatenate(axis = 3)([encoder_output, image_feature])
            encoder_output = layer(fusion_output)

        decoder_output = encoder_output
        for layer in decoder_layers:
            decoder_output = layer(decoder_output)

        def image_loss(y_true, y_pred):
            return tf.norm(y_true - y_pred)

        # tie it together [image, seq] [word]
        self.model = Model(inputs=[inputs, mask_inputs, hue_inputs], outputs=decoder_output)
        self.model.compile(
            loss=[image_loss], optimizer='adam', metrics=['acc']
        )
        # summarize model
        print(self.model.summary())
        # plot_model(self.model, to_file='autoencoder_colorization_merged.png', show_shapes=True)

    def train(self, image_dir, data_dir, num_train_samples, batch_size, **kwargs):
        dataset = HairColorizationPyDataset(
            image_dir, data_dir, num_train_samples, batch_size,
            self.img_h, self.img_w,
            **kwargs
        )
        steps_per_epoch = np.uint8(np.floor(num_train_samples / batch_size))

        self.fit_history = self.model.fit(dataset, epochs=3, steps_per_epoch=steps_per_epoch, verbose=1)

        self.model.save('model_merge.h5')

def preprocess_image(img_h, img_w, img, gray_img, b_mask):
    window_left_top_y, window_left_top_x = img.shape[0], img.shape[1]
    window_right_bottom_y, window_right_bottom_x = 0, 0
    for y, mask_y in enumerate(b_mask):
        for x, mask in enumerate(mask_y):
            if mask.any():
                window_left_top_y = min(window_left_top_y, y)
                window_left_top_x = min(window_left_top_x, x)
                window_right_bottom_y = max(window_right_bottom_y, y)
                window_right_bottom_x = max(window_right_bottom_x, x)

    expected_img = img[window_left_top_y:window_right_bottom_y + 1,window_left_top_x: window_right_bottom_x + 1][:]
    processed_img = expected_img.copy()
    mask_shape = (window_right_bottom_y - window_left_top_y + 1, window_right_bottom_x - window_left_top_x + 1, 1)
    processed_mask = np.zeros(mask_shape, np.uint8)

    for y in range(window_left_top_y, window_right_bottom_y + 1):
        for x in range(window_left_top_x, window_right_bottom_x + 1):
            if b_mask[y][x].any():
                processed_img[y - window_left_top_y][x - window_left_top_x] = gray_img[y][x]
                processed_mask[y - window_left_top_y][x - window_left_top_x] = [255]

    processed_img = cv.resize(processed_img, (img_h, img_w), interpolation= cv.INTER_LINEAR)
    processed_mask = cv.resize(processed_mask, (img_h, img_w), interpolation= cv.INTER_LINEAR)
    processed_mask = processed_mask.reshape((img_h, img_w, 1))
    expected_img = cv.resize(expected_img, (img_h, img_w), interpolation= cv.INTER_LINEAR)

    log(1, "processed_img shape: {0}", processed_img.shape)
    log(1, "processed_mask shape: {0}", processed_mask.shape)
    log(1, "expected_img shape: {0}", expected_img.shape)
    return processed_img, processed_mask, expected_img

class HairColorizationPyDataset(PyDataset):

    def __init__(
        self,
        image_dir,
        data_dir,
        num_train_samples,
        batch_size,
        img_h, img_w,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert batch_size % 4 == 0, "batch_size must always be divisible to 4."
        self.data_list = glob.glob(path.abspath(data_dir) + f'/*.json')
        assert len(self.data_list) >= num_train_samples, "len of data_list: {0}, num_train_samples: {1}".format(self.data_list, num_train_samples)
        self.image_dir, self.data_dir = path.abspath(image_dir), path.abspath(data_dir)
        self.num_train_samples, self.batch_size = num_train_samples, batch_size
        self.img_h, self.img_w = img_h, img_w
        self.current_idx = 0

    def __len__(self):
        # Return number of batches.
        return math.ceil(self.num_train_samples / self.batch_size)

    def __getitem__(self, idx):
        data_list = self.data_list[self.current_idx:]

        true_img_list, processed_img_list, mask_list, hue_list = list(), list(), list(), list()
        for json_path in data_list:
            json_name = path.basename(json_path)
            splitted_name = path.splitext(json_name)
            image_name = splitted_name[0] + '.jpg'
            gray_name = splitted_name[0] + '-gray-hair.jpg'
            b_mask_name = splitted_name[0] + '-b-mask.png'
            image_path = os.path.join(self.image_dir, image_name)
            gray_path = os.path.join(self.data_dir, gray_name)
            b_mask_path = os.path.join(self.data_dir, b_mask_name)

            hues = []
            with open(json_path, 'r') as f:
                try:
                    data = json.loads(f.read())
                    hues.append(data['min_hue'])
                    hues.append(data['max_hue'])
                    hues.append(data['mean_hue'])
                    hues.append(data['median_hue'])
                except json.JSONDecodeError as e:
                    log(2, "cannot parse: {0}, {1}", json_path, e)
                    continue
            assert(len(hues) == 4)

            img = cv.imread(image_path)
            if img is None:
                log(2, "cannot parse image: {0}", image_path)
                continue

            gray_img = cv.imread(gray_path)
            if gray_img is None:
                log(2, "cannot parse gray image: {0}", gray_path)
                continue

            b_mask = cv.imread(b_mask_path)
            if b_mask is None:
                log(2, "cannot parse b-mask: {0}", b_mask_path)
                continue

            processed_img, processed_mask, true_img = preprocess_image(
                self.img_h, self.img_w, img, gray_img, b_mask
            )

            for hue in hues:
                hue_list.append(hue)
                processed_img_list.append(processed_img.copy())
                mask_list.append(processed_mask.copy())
                true_img_list.append(true_img.copy())

            self.current_idx += 1
            if len(processed_img_list) >= self.batch_size:
                break

        processed_img_list = np.array(processed_img_list)
        mask_list = np.array(mask_list)
        hue_list = np.array(hue_list)
        true_img_list = np.array(true_img_list)
        log(2, "processed_img_list shape: {0}", processed_img_list.shape)
        log(2, "mask_list shape: {0}", mask_list.shape)
        log(2, "hue_list shape: {0}", hue_list.shape)
        log(2, "img_true_list shape: {0}", true_img_list.shape)
        return (
            (processed_img_list, mask_list, hue_list),
            true_img_list,
        )


def log(level, s, *arg):
     if level > 2:
        if arg:
            print(s.format(*arg))
        else:
            print(s)


