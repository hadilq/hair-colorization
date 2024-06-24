import tensorflow as tf
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Lambda
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
from random import randint

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(dir_path + '/../model/hair_colorization.keras')

class HairColorization:

    def __init__(self, img_h, img_w):
        self.img_h, self.img_w = img_h, img_w

    def define_model_step_1(self):
        initializer = tf.keras.initializers.HeNormal()

        encoder_layers = [
            Conv2D(
                128, (3,3), activation='relu',
                kernel_initializer=initializer, padding='same', strides=2,
                name = 'step_1_en_conv2d_128'
            ),
            Conv2D(
                256, (3,3), activation='relu',
                kernel_initializer=initializer, padding='same', strides=2,
                name = 'step_1_en_conv2d_256'
            ),
            BatchNormalization(axis=3, name = 'step_1_en_batch_normalization'),
            Activation("softmax"),
        ]

        decoder_layers = [
            Conv2D(
                128, (3,3), activation='relu',
                kernel_initializer=initializer, padding='same',
                name = 'step_1_de_conv2d_128'
            ),
            UpSampling2D((2, 2)),
            Conv2D(
                3, (3, 3), activation='relu',
                kernel_initializer=initializer, padding='same',
                name = 'step_1_de_conv2d_3'
            ),
            UpSampling2D((2, 2)),
            Activation("softmax"),
        ]

        self.build_model(encoder_layers, decoder_layers)

    def define_model_step_2_dry_run(self):
        img_inputs = Input(shape=(self.img_h, self.img_w, 3,))
        hue_inputs = Input(shape=(1,))

        initializer = tf.keras.initializers.HeNormal()

        encoder_layers = [
            Conv2D(
                128, (3,3), activation='relu',
                kernel_initializer=initializer, padding='same',
                name = 'step_2_en_conv2d_128'
            ),
            Conv2D(
                128, (3,3), activation='relu',
                kernel_initializer=initializer, padding='same', strides=2,
                name = 'step_1_en_conv2d_128'
            ),
            Conv2D(
                256, (3,3), activation='relu',
                kernel_initializer=initializer, padding='same',
                name = 'step_2_en_conv2d_256'
            ),
            Conv2D(
                256, (3,3), activation='relu',
                kernel_initializer=initializer, padding='same', strides=2,
                name = 'step_1_en_conv2d_256'
            ),
            Conv2D(
                512, (3,3), activation='relu',
                kernel_initializer=initializer, padding='same', strides=2,
                name = 'step_2_en_conv2d_512'
            ),
            BatchNormalization(axis=3, name = 'step_1_en_batch_normalization'),
            Activation("softmax"),
        ]

        decoder_layers = [
            Conv2D(
                256, (3, 3), activation='relu',
                kernel_initializer=initializer, padding='same',
                name = 'step_2_de_conv2d_256'
            ),
            UpSampling2D((2, 2)),
            Conv2D(
                128, (3,3), activation='relu',
                kernel_initializer=initializer, padding='same',
                name = 'step_1_de_conv2d_128'
            ),
            UpSampling2D((2, 2)),
            Conv2D(
                3, (3, 3), activation='relu',
                kernel_initializer=initializer, padding='same',
                name = 'step_1_de_conv2d_3'
            ),
            UpSampling2D((2, 2)),
            Activation("softmax"),
        ]

        self.build_model(img_inputs, hue_inputs, encoder_layers, decoder_layers)

    def define_model_step_2(self):
        """
            To run this step, you need to already defined the model in the step one,
            the modify it add more layers.
        """
        initializer = tf.keras.initializers.HeNormal()

        layers = self.model.layers
        encoder_layers_count = 18
        encoder_layers = [l for l in layers[:encoder_layers_count] if l.weights]
        decoder_layers = [l for l in layers[encoder_layers_count:] if l.weights]

        log(3, "encoder_layers with weights: {0}", encoder_layers)
        log(3, "decoder_layers with weights: {0}", decoder_layers)
        
        encoder_layers_insert_map = {
            1: [
                Conv2D(
                    128, (3,3), activation='relu',
                    kernel_initializer=initializer, padding='same',
                    name = 'step_2_en_conv2d_128'
                ),
            ],
            2: [
                Conv2D(
                    256, (3,3), activation='relu',
                    kernel_initializer=initializer, padding='same',
                    name = 'step_2_en_conv2d_256'
                ),
            ],
            3: [ Activation("softmax") ],
        }

        decoder_layers_insert_map = {
            0: [
                Conv2D(
                    256, (3, 3), activation='relu',
                    kernel_initializer=initializer, padding='same',
                    name = 'step_2_de_conv2d_256'
                ),
                UpSampling2D((2, 2)),
                Lambda(
                    lambda x: tf.pad(
                        x,
                        tf.constant([[0, 0], [0, 0], [0, 0], [0, 2]]),
                        'CONSTANT',
                    ),
                    output_shape=(120, 120, 258),
                ),
            ],
            2: [
                UpSampling2D((2, 2)),
                Activation("softmax"),
            ],
        }

        step_2_encoder_layers = self.insert_layers(encoder_layers, encoder_layers_insert_map)
        step_2_decoder_layers = self.insert_layers(decoder_layers, decoder_layers_insert_map)

        log(3, "new_encoder_layers: {0}", step_2_encoder_layers)
        log(3, "new_decoder_layers: {0}", step_2_decoder_layers)
        self.build_model(step_2_encoder_layers, step_2_decoder_layers)

    def insert_layers(self, old_layers, new_layers_insert_map):
        count = 0
        updated_layers = []
        while count < len(old_layers):
            if count in new_layers_insert_map:
                updated_layers.extend(new_layers_insert_map[count])
                del new_layers_insert_map[count]
            else:
                updated_layers.append(old_layers[count])
                count += 1

        rest_keys = sorted(new_layers_insert_map.keys())
        for k in rest_keys:
            updated_layers.extend(new_layers_insert_map[k])
        return updated_layers


    def build_model(self, encoder_layers, decoder_layers):
        img_inputs = Input(shape=(self.img_h, self.img_w, 3,))
        hue_inputs = Input(shape=(1,))

        encoder_output = img_inputs
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

        self.model = Model(inputs=[img_inputs, hue_inputs], outputs=decoder_output)
        self.compile_model()
        # summarize model
        self.model.summary()

        # Un-comment before commit!!
        # plot_model(self.model, to_file=dir_path + f'/autoencoder_colorization.png', show_shapes=True)

    def load_model_step_1(self, model_path=model_path):
        self.define_model_step_1()
        self.model.load_weights(model_path)

    def load_model_step_2(self, model_path=model_path):
        self.define_model_step_1()
        self.define_model_step_2()
        self.model.load_weights(model_path)

    def load_model(self, model_path=model_path):
        self.model = tf.keras.models.load_model(model_path, compile=False)
        self.compile_model()

    def train(
            self, image_dir, output_dir, data_dir,
            num_train_samples, batch_size, model_path=model_path,
            init_index=3,
            **kwargs
        ):
        dataset = HairColorizationPyDataset(
            image_dir, output_dir, data_dir, num_train_samples, batch_size,
            self.img_h, self.img_w,
            self.preprocess_image,
            **kwargs
        )

        index = init_index
        while True:
            random_int = randint(1, 1000000)
            tf.keras.utils.set_random_seed(random_int)
            tf.config.experimental.enable_op_determinism()
            log(3, "random seed: {0}", random_int)

            self.fit_history = self.model.fit(dataset, epochs=10, verbose=2)

            self.save_model_weights(model_path, index)
            index += 1

            log(3, "history: {0}", self.fit_history.history)
            max_accuracy = max(self.fit_history.history.get('acc'))
            log(3, "max accuracy: {0}", max_accuracy)
            if max_accuracy > 0.95:
                break

    def save_model(self, model_path, index):
        model_name = path.basename(model_path)
        splitted_name = path.splitext(model_name)
        indexed_model_path = os.path.join(
            os.path.dirname(model_path),
            splitted_name[0] + '.' + str(index) + splitted_name[1]
        )
        self.model.save(indexed_model_path)

    def save_model_weights(self, model_path, index):
        model_name = path.basename(model_path)
        splitted_name = path.splitext(model_name)
        indexed_model_path = os.path.join(
            os.path.dirname(model_path),
            splitted_name[0] + '.' + str(index) + '.weights.h5'
        )
        self.model.save_weights(indexed_model_path)

    def compile_model(self):
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            #optimizer=SGD(
            #    learning_rate=0.1
            #),
            #loss='mean_squared_error',
            loss='categorical_crossentropy',
            metrics=['acc']
        )

    def predict(self, img, b_mask, gray_img, hue, output_dir):
        """
            Evaluate the mode to predict.
            The 'img', and 'gray_img' must be in HSV space.
            The outputs are also in HSV space.
        """
        if img is None:
            log(3, "cannot make data!")
            return None
        processed_img, true_img, window = self.preprocess_image(
                img, hue, gray_img, b_mask, output_dir
        )
        logImage(3, output_dir, processed_img)
        logImage(3, output_dir, true_img)
        prediction = self.model.predict(
            ( np.array([ processed_img ]), np.array([ hue ]) ),
            verbose = 2
        )
        log(3, "predicted shape: {0}", prediction.shape)
        return self.postprocess_image(
            img, (prediction[0]).astype(np.uint8), b_mask, window, output_dir
        )

    def postprocess_image(self, img, prediction, b_mask, window, output_dir):
        """
            Postprocess image. The 'img' must be in HSV space. The outputs are also in HSV space.
        """
        logImage(3, output_dir, prediction)
        log(3, "predicted image: {0}", prediction)
        window_top_left_y, window_top_left_x,\
            window_bottom_right_y, window_bottom_right_x = window
        window_size = (
            window_bottom_right_y - window_top_left_y,
            window_bottom_right_x - window_top_left_x,
        )
        processed_img = cv.resize(prediction, window_size, interpolation= cv.INTER_LINEAR)
        logImage(3, output_dir, processed_img)

        postprocess_image = img.copy()
        log(
            3, "shape of processed_img: {0}, postprocess_image: {1}, window: {2}",
            processed_img.shape, postprocess_image.shape, window
        )
        for y in range(window_top_left_y, window_bottom_right_y):
            for x in range(window_top_left_x, window_bottom_right_x):
                if b_mask[y][x].any():
                    postprocess_image[y][x] = processed_img[y - window_top_left_y][x - window_top_left_x]

        return postprocess_image

    def preprocess_image(self, img, requested_hue, gray_img, b_mask, output_dir):
        """
            Preprocess image. The 'img' must be in HSV space. The outputs are also in HSV space.
        """
        assert self.img_h / self.img_w == img.shape[0] / img.shape[1],\
            "the requested width/height ratio must be the same as ratio in the original image."
        assert self.img_h < img.shape[0],\
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
        expected_image_size_ratio = self.img_h / self.img_w
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
        processed_img = gray_img[window_top_left_y:window_bottom_right_y + 1,window_top_left_x: window_bottom_right_x + 1][:]
        expected_img = expected_img.copy()
        processed_img = processed_img.copy()
        log(ll, "expected_img shape: {0}", expected_img.shape)
        log(ll, "processed_img shape: {0}", processed_img.shape)

        processed_img = cv.resize(processed_img, (self.img_h, self.img_w), interpolation= cv.INTER_LINEAR)
        expected_img = cv.resize(expected_img, (self.img_h, self.img_w), interpolation= cv.INTER_LINEAR)

        logImage(1, output_dir, img)
        logImage(1, output_dir, expected_img)
        logImage(1, output_dir, processed_img)

        log(1, "processed_img shape: {0}", processed_img.shape)
        log(1, "expected_img shape: {0}", expected_img.shape)
        window = (
            window_top_left_y,
            window_top_left_x,
            window_bottom_right_y,
            window_bottom_right_x,
        )

        return processed_img, expected_img, window

class HairColorizationPyDataset(PyDataset):

    def __init__(
        self,
        image_dir,
        output_dir,
        data_dir,
        num_train_samples,
        batch_size,
        img_h, img_w,
        preprocess_image_fn,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sample_expansion_rate = 3 # it's defined by the number of data points.
        assert batch_size % self.sample_expansion_rate == 0,\
                "batch_size({0}) must be divisible by {1}".format(batch_size, self.sample_expansion_rate)
        self.data_list = glob.glob(path.abspath(data_dir) + f'/*.json')
        assert len(self.data_list) >= num_train_samples,\
                "length of data_list: {0}, num_train_samples: {1}".format(self.data_list, num_train_samples)
        self.output_dir = path.abspath(output_dir)
        self.image_dir, self.data_dir = path.abspath(image_dir), path.abspath(data_dir)
        self.num_train_samples, self.batch_size = num_train_samples, batch_size
        self.img_h, self.img_w = img_h, img_w
        self.preprocess_image = preprocess_image_fn

    def __len__(self):
        # return number of batches.
        return np.uint8(np.floor(self.num_train_samples * self.sample_expansion_rate / self.batch_size))

    def __getitem__(self, idx):
        data_idx = idx * self.batch_size // self.sample_expansion_rate
        ll = 2

        true_img_list, processed_img_list, hue_list = list(), list(), list()
        while len(processed_img_list) < self.batch_size and data_idx < len(self.data_list):
            json_path = self.data_list[data_idx]
            data_idx += 1

            hues = []
            with open(json_path, 'r') as f:
                try:
                    data = json.loads(f.read())
                    # hues.append(data['min_hue'])
                    # hues.append(data['max_hue'])
                    hues.append(data['mean_hue'])
                    hues.append(data['median_hue'])
                    hues.append(data['mode_hue'])
                except json.JSONDecodeError as e:
                    raise "cannot parse: {0}, {1}".format(json_path, e)
            # This is how we define the self.sample_expansion_rate
            assert(len(hues) == self.sample_expansion_rate)

            json_name = path.basename(json_path)
            splitted_name = path.splitext(json_name)
            processed_img_path_list, true_img_path_list = list(), list()
            all_caches_exist = True
            for i in range(len(hues)):
                processed_img_name = splitted_name[0] + '-processed-img-3-cache.' + str(i) + '.jpg'
                true_img_name = splitted_name[0] + '-true-img-cache.1.' + str(i) + '.jpg'
                processed_img_path = os.path.join(self.data_dir, processed_img_name)
                true_img_path = os.path.join(self.data_dir, true_img_name)
                processed_img_path_list.append(processed_img_path)
                true_img_path_list.append(true_img_path)
                all_caches_exist = all_caches_exist and os.path.exists(processed_img_path)\
                    and os.path.exists(true_img_path)

            if all_caches_exist:
                for i in range(len(hues)):
                    processed_img_path = processed_img_path_list[i]
                    true_img_path = true_img_path_list[i]
                    processed_img = cv.imread(processed_img_path)
                    true_img = cv.imread(true_img_path)
                    if processed_img is not None and true_img is not None:
                        processed_img = cv.cvtColor(processed_img, cv.COLOR_BGR2HSV)
                        true_img = cv.cvtColor(true_img, cv.COLOR_BGR2HSV)
                        hue_list.append(hues[i])
                        processed_img_list.append(processed_img)
                        true_img_list.append(true_img)
                    else:
                        if processed_img is None:
                            log(4, "cannot read the processed_img cache! {0}", processed_img_path)
                        if true_img is None:
                            log(4, "cannot read the true_img cache! {0}", true_img_path)
                        raise "couldn't read cache file! {0}, {1}".format(processed_img_path, true_img_path)
                continue
            else:
                log(4, "No cache for {0}!", json_path)

            image_name = splitted_name[0] + '.jpg'
            gray_name = splitted_name[0] + '-gray-hair.jpg'
            b_mask_name = splitted_name[0] + '-b-mask.png'
            image_path = os.path.join(self.image_dir, image_name)
            gray_path = os.path.join(self.data_dir, gray_name)
            b_mask_path = os.path.join(self.data_dir, b_mask_name)

            img = cv.imread(image_path)
            if img is None:
                raise "cannot parse: {0}".format(image_path)
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            gray_img = cv.imread(gray_path)
            if gray_img is None:
                log(3, )
                raise "cannot parse gray image: {0}".format(gray_path)
            gray_img = cv.cvtColor(gray_img, cv.COLOR_BGR2HSV)

            b_mask = cv.imread(b_mask_path)
            if b_mask is None:
                log(3, )
                raise "cannot parse b-mask: {0}".format(b_mask_path)

            for i in range(len(hues)):
                hue = hues[i]
                processed_img, true_img, _ = self.preprocess_image(
                    img, hue, gray_img, b_mask, self.output_dir
                )

                processed_img_path = processed_img_path_list[i]
                true_img_path = true_img_path_list[i]
                log(ll, "processed_img_path: {0}", processed_img_path)
                log(ll, "true_img_path: {0}", true_img_path)
                cv.imwrite(processed_img_path, cv.cvtColor(processed_img, cv.COLOR_HSV2BGR))
                cv.imwrite(true_img_path,cv.cvtColor(true_img, cv.COLOR_HSV2BGR))

                hue_list.append(hue)
                processed_img_list.append(processed_img)
                true_img_list.append(true_img)

            log(ll, "len processed_img_list: {0}", len(processed_img_list))
            log(ll, "image_name: {0}", image_name)
            log(ll, "processed_img shape: {0}", processed_img.shape)
            log(ll, "true_img shape: {0}", true_img.shape)

        processed_img_list = np.array(processed_img_list)
        hue_list = np.array(hue_list)
        true_img_list = np.array(true_img_list)
        log(ll, "processed_img_list shape: {0}", processed_img_list.shape)
        log(ll, "hue_list shape: {0}", hue_list.shape)
        log(ll, "img_true_list shape: {0}", true_img_list.shape)
        log(ll, "idx: {0}", idx)
        log(ll, "data_idx: {0}", data_idx)
        log(ll, "len data_list: {0}", len(self.data_list))
        assert processed_img_list.shape[0] != 0,\
                "processed_img_list is empty! __len__: {0}, len data_list: {1}, current_idx: {2}, idx: {3}"\
                .format(self.__len__(), len(self.data_list), data_idx, idx)
        return (
            (processed_img_list, hue_list),
            true_img_list,
        )

# Helper function to both segment the hairs and color them.
def predict(image_path, hue, hair_sgment_model_path, hair_colorization_model_path, output_dir):
    import gc
    import importlib
    import sys
    import torch
    torch.cuda.empty_cache()
    sys.path.append(dir_path)
    segment_mod = importlib.import_module('hair_segment')
    segmentor = segment_mod.HairSegment()
    segmentor.load_model(hair_sgment_model_path)
    img, b_mask, _ = segmentor.make_data(image_path)
    gray_img = segmentor.make_gray_hair(img, b_mask)
    img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
    gray_img = cv.cvtColor(cv.cvtColor(gray_img, cv.COLOR_GRAY2BGR), cv.COLOR_RGB2HSV)
    del segmentor
    # the trained model expects 240x240 image size
    colorizor = HairColorization(240, 240)
    colorizor.load_model_step_2(hair_colorization_model_path)
    result = colorizor.predict(img, b_mask, gray_img, hue, output_dir)
    del colorizor

    logImage(3, output_dir, img)
    result = cv.cvtColor(result, cv.COLOR_HSV2BGR)
    logImage(3, output_dir, result, img_space=LOG_BGR_IMAGE)
    return result


"""
Image spaces used by logger.
"""
LOG_HSV_IMAGE = 0
LOG_BGR_IMAGE = 1

def logImage(level, output_dir, img, img_space=LOG_HSV_IMAGE):
    """
        Log image in Jupyter lab.
    """
    if level > 2:
       from IPython.display import display, Image

       img_path = output_dir + f'/.log_img.png'
       if img_space == LOG_HSV_IMAGE:
           img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
       cv.imwrite(img_path, img)
       display(Image(filename=img_path, height=600))

def log(level, s, *arg):
    if level > 2:
       if arg:
           print(s.format(*arg))
       else:
           print(s)


