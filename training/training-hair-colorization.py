import tensorflow as tf
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.layers import Flatten
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

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = dir_path + f'/model.tf'

class HairColorizationTraining:

    def __init__(self, img_h, img_w):
        self.img_h, self.img_w = img_h, img_w

    def define_model(self):
        img_inputs = Input(shape=(self.img_h, self.img_w, 3,))
        mask_inputs = Input(shape=(self.img_h, self.img_w, 1,))
        hue_inputs = Input(shape=(1,))

        encoder_layers = [
            Conv2D(64, (3,3), activation='relu', padding='same', strides=2),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            Conv2D(128, (3,3), activation='relu', padding='same', strides=2),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same', strides=2),
            Conv2D(512, (3,3), activation='relu', padding='same'),
            Conv2D(512, (3,3), activation='relu', padding='same'),
            Conv2D(512, (3,3), activation='relu', padding='same'),
            Conv2D(512, (3,3), activation='relu', padding='same'),
            Conv2D(512, (3,3), activation='relu', padding='same'),
            Conv2D(512, (3,3), activation='relu', padding='same', strides=2),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same'),
        ]

        decoder_layers = [
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            Conv2D(256, (3,3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            Conv2D(128, (3,3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            Conv2D(64, (3,3), activation='relu', padding='same'),
            UpSampling2D((2, 2)),
            Conv2D(32, (3,3), activation='relu', padding='same'),
            Conv2D(32, (3,3), activation='relu', padding='same'),
            Conv2D(32, (3,3), activation='relu', padding='same'),
            Conv2D(32, (3,3), activation='relu', padding='same'),
            Conv2D(16, (3,3), activation='relu', padding='same'),
            Conv2D(3, (3, 3), activation='tanh', padding='same'),
            UpSampling2D((2, 2)),
        ]

        encoder_output = Concatenate(axis = 3)([img_inputs, mask_inputs])
        for layer in encoder_layers:
            concat_shape = (np.uint32(encoder_output.shape[1]), np.uint32(encoder_output.shape[2]), np.uint32(hue_inputs.shape[-1]))

            image_feature = RepeatVector((concat_shape[0] * concat_shape[1]).item())(hue_inputs)
            image_feature = Reshape(concat_shape)(image_feature)
            fusion_output = Concatenate(axis = 3)([encoder_output, image_feature])
            encoder_output = layer(fusion_output)

        decoder_output = encoder_output
        for layer in decoder_layers:
            decoder_output = layer(decoder_output)

        assert decoder_output.shape == img_inputs.shape,\
            "decoder_output shape is {0}, while inputs shape is {1}. They should be the same!"\
            .format(decoder_output.shape, img_inputs.shape)

        self.model = Model(inputs=[img_inputs, mask_inputs, hue_inputs], outputs=decoder_output)
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001), loss=categorical_crossentropy,
            metrics=['accuracy']
        )
        # summarize model
        self.model.summary()

        # Un-comment before commit!!
        plot_model(self.model, to_file=dir_path + f'/autoencoder_colorization.png', show_shapes=True)

    def train(self, image_dir, output_dir, data_dir, num_train_samples, batch_size, model_path=model_path, **kwargs):
        dataset = HairColorizationPyDataset(
            image_dir, output_dir, data_dir, num_train_samples, batch_size,
            self.img_h, self.img_w,
            **kwargs
        )
        steps_per_epoch = np.uint8(np.floor(num_train_samples / batch_size))

        self.fit_history = self.model.fit(dataset, epochs=10, steps_per_epoch=steps_per_epoch, verbose=2)

        self.model.save(model_path)

    def load_model(self, model_path=model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, segmentor, image_path, hue, output_dir):
        import importlib
        img, b_mask, _ = segmentor.make_data(image_path)
        if img is None:
            log(3, "cannot make data!")
            return None
        gray_img = segmentor.make_gray_hair(img, b_mask)
        processed_img, processed_mask, _ = preprocess_image(
                240, 240, img, gray_img, b_mask, output_dir
        )
        return model.predict(
            (( np.array([ processed_img ]),
               np.array([ processed_mask ]),
               np.array([ hue ])
              )),
            verbose = 2
        )

class HairColorizationPyDataset(PyDataset):

    def __init__(
        self,
        image_dir,
        output_dir,
        data_dir,
        num_train_samples,
        batch_size,
        img_h, img_w,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert batch_size % 4 == 0, "batch_size must always be divisible to 4."
        self.data_list = glob.glob(path.abspath(data_dir) + f'/*.json')
        assert 4 * len(self.data_list) >= num_train_samples, "len of data_list: {0}, num_train_samples: {1}".format(self.data_list, num_train_samples)
        self.output_dir = path.abspath(output_dir)
        self.image_dir, self.data_dir = path.abspath(image_dir), path.abspath(data_dir)
        self.num_train_samples, self.batch_size = num_train_samples, batch_size
        self.img_h, self.img_w = img_h, img_w

    def __len__(self):
        # return number of batches.
        return math.ceil(self.num_train_samples / self.batch_size)

    def __getitem__(self, idx):
        data_idx = idx * self.batch_size // 4
        ll = 2

        true_img_list, processed_img_list, mask_list, hue_list = list(), list(), list(), list()
        while len(processed_img_list) < self.batch_size and data_idx < len(self.data_list):
            json_path = self.data_list[data_idx]
            data_idx += 1
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
                    log(3,)
                    raise "cannot parse: {0}, {1}".format(json_path, e)
            assert(len(hues) == 4)

            img = cv.imread(image_path)
            if img is None:
                raise "cannot parse: {0}".format(image_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

            gray_img = cv.imread(gray_path)
            if gray_img is None:
                log(3, )
                raise "cannot parse gray image: {0}".format(gray_path)

            b_mask = cv.imread(b_mask_path)
            if b_mask is None:
                log(3, )
                raise "cannot parse b-mask: {0}".format(b_mask_path)

            processed_img, processed_mask, true_img = preprocess_image(
                self.img_h, self.img_w, img, gray_img, b_mask, self.output_dir
            )

            for hue in hues:
                hue_list.append(hue)
                processed_img_list.append(processed_img.copy())
                mask_list.append(processed_mask.copy())
                true_img_list.append(true_img.copy())

            log(ll, "len processed_img_list: {0}", len(processed_img_list))
            log(ll, "image_name: {0}", image_name)
            log(ll, "processed_img shape: {0}", processed_img.shape)
            log(ll, "processed_mask shape: {0}", processed_mask.shape)
            log(ll, "true_img shape: {0}", true_img.shape)

        processed_img_list = np.array(processed_img_list)
        mask_list = np.array(mask_list)
        hue_list = np.array(hue_list)
        true_img_list = np.array(true_img_list)
        log(ll, "processed_img_list shape: {0}", processed_img_list.shape)
        log(ll, "mask_list shape: {0}", mask_list.shape)
        log(ll, "hue_list shape: {0}", hue_list.shape)
        log(ll, "img_true_list shape: {0}", true_img_list.shape)
        log(ll, "idx: {0}", idx)
        log(ll, "data_idx: {0}", data_idx)
        log(ll, "len data_list: {0}", len(self.data_list))
        assert processed_img_list.shape[0] != 0,\
                "processed_img_list is empty! __len__: {0}, len data_list: {1}, current_idx: {2}, idx: {3}"\
                .format(self.__len__(), len(self.data_list), data_idx, idx)
        return (
            (processed_img_list, mask_list, hue_list),
            true_img_list,
        )

def preprocess_image(img_h, img_w, img, gray_img, b_mask, output_dir):
    assert img_h / img_w == img.shape[0] / img.shape[1],\
        "the requested width/height ratio must be the same as ratio in the original image."
    assert img_h < img.shape[0],\
        "the requested width/height must be smaller than the original image."
    window_top_left_y, window_top_left_x = img.shape[0], img.shape[1]
    window_bottom_right_y, window_bottom_right_x = 0, 0
    ll = 1
    for y, mask_y in enumerate(b_mask):
        for x, mask in enumerate(mask_y):
            if mask.any():
                window_top_left_y = min(window_top_left_y, y)
                window_top_left_x = min(window_top_left_x, x)
                window_bottom_right_y = max(window_bottom_right_y, y)
                window_bottom_right_x = max(window_bottom_right_x, x)

    log(ll, "before window_top_left_y: {0}", window_top_left_y)
    log(ll, "before window_top_left_x: {0}", window_top_left_x)
    log(ll, "before window_bottom_right_y: {0}", window_bottom_right_y)
    log(ll, "before window_bottom_right_x: {0}", window_bottom_right_x)

    # Keep the ratio of the window
    window_size = [window_bottom_right_y - window_top_left_y, window_bottom_right_x - window_top_left_x]
    window_center = [window_top_left_y + window_size[0] / 2, window_top_left_x + window_size[1] / 2]
    expected_image_size_ratio = img_h / img_w
    log(ll, "img shape: {0}", img.shape)
    log(ll, "expected_image_size_ratio: {0}", expected_image_size_ratio)
    log(ll, "before window_size: {0}", window_size)
    log(ll, "before window_center: {0}", window_center)
    if window_size[0] > window_size[1]:
        new_half_size_1 = (window_size[0] / expected_image_size_ratio) / 2
        window_size[1] = window_size[0] / expected_image_size_ratio
        if window_center[1] + new_half_size_1 >= img.shape[1]:
            window_center[1] = img.shape[1] - new_half_size_1 - 1
        elif window_center[1] - new_half_size_1 < 0:
            window_center[1] = new_half_size_1
    else:
        new_half_size_0 = (window_size[1] * expected_image_size_ratio) / 2
        window_size[0] = window_size[1] * expected_image_size_ratio
        if window_center[0] + new_half_size_0 >= img.shape[0]:
            window_center[0] = img.shape[0] - new_half_size_0 - 1
        elif window_center[0] - new_half_size_0 < 0:
            window_center[0] = new_half_size_0

    log(ll, "after window_size: {0}", window_size)
    log(ll, "after window_center: {0}", window_center)

    # Apply the adjusted window to the ratio
    window_top_left_y = int(window_center[0] - (window_size[0] // 2))
    window_top_left_x = int(window_center[1] - (window_size[1] // 2))
    window_bottom_right_y = int(window_center[0] + (window_size[0] // 2))
    window_bottom_right_x = int(window_center[1] + (window_size[1] // 2))

    log(ll, "after window_top_left_y: {0}", window_top_left_y)
    log(ll, "after window_top_left_x: {0}", window_top_left_x)
    log(ll, "after window_bottom_right_y: {0}", window_bottom_right_y)
    log(ll, "after window_bottom_right_x: {0}", window_bottom_right_x)

    expected_img = img[window_top_left_y:window_bottom_right_y + 1,window_top_left_x: window_bottom_right_x + 1][:]
    processed_img = expected_img.copy()
    mask_shape = (window_bottom_right_y - window_top_left_y + 1, window_bottom_right_x - window_top_left_x + 1, 1)
    processed_mask = np.zeros(mask_shape, np.uint8)
    log(ll, "expected_img shape: {0}", expected_img.shape)
    log(ll, "processed_img shape: {0}", processed_img.shape)
    log(ll, "processed_mask shape: {0}", processed_mask.shape)

    for y in range(window_top_left_y, window_bottom_right_y + 1):
        for x in range(window_top_left_x, window_bottom_right_x + 1):
            if b_mask[y][x].any():
                processed_img[y - window_top_left_y][x - window_top_left_x] = gray_img[y][x]
                processed_mask[y - window_top_left_y][x - window_top_left_x] = [255]

    processed_img = cv.resize(processed_img, (img_h, img_w), interpolation= cv.INTER_LINEAR)
    processed_mask = cv.resize(processed_mask, (img_h, img_w), interpolation= cv.INTER_LINEAR)
    processed_mask = processed_mask.reshape((img_h, img_w, 1))
    expected_img = cv.resize(expected_img, (img_h, img_w), interpolation= cv.INTER_LINEAR)

    logImage(1, output_dir, img)
    logImage(1, output_dir, expected_img)
    logImage(1, output_dir, processed_img)

    log(1, "processed_img shape: {0}", processed_img.shape)
    log(1, "processed_mask shape: {0}", processed_mask.shape)
    log(1, "expected_img shape: {0}", expected_img.shape)
    return processed_img, processed_mask, expected_img


def logImage(level, output_dir, img):
     if level > 2:
        from PIL import Image as Img
        from IPython.display import display, Image

        img_path = output_dir + f'/img.png'
        Img.fromarray(img, 'RGB').save(img_path)
        display(Image(filename=img_path, height=600))

def log(level, s, *arg):
     if level > 2:
        if arg:
            print(s.format(*arg))
        else:
            print(s)


