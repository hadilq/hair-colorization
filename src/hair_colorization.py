import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from torchvision import models
import torchvision.transforms.functional as TF
import os
import glob
import math
import json
from os import path
import numpy as np
import cv2 as cv
import importlib
from importlib import reload

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.abspath(dir_path + '/../model/hair_colorization.pt')

class MaskedLoss(nn.Module):
    def __init__(self):
        super(MaskedLoss, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, pred, target, input_mask):
        mse = F.mse_loss(pred, target, reduction='none')

        # Apply the mask
        masked_mse = mse * input_mask

        # Calculate the mean loss over the masked elements
        num_valid = input_mask.sum().clamp(min=1e-8)
        loss = masked_mse.sum() / num_valid

        return loss

class MaskedLayer(nn.Module):
    def __init__(self):
        super(MaskedLayer, self).__init__()

    def forward(self, x, img_inputs):
        mask = img_inputs[:, 1:2, :, :]
        mask = mask.repeat(1, 2, 1, 1)
        true_255 = (mask == 255)
        masked_output = torch.where(true_255, x, torch.tensor(0.0, device=x.device))
        return masked_output

class HairColorizationModel(nn.Module):
    def __init__(self, img_h, img_w, processed_img_depth):
        super(HairColorizationModel, self).__init__()
        self.img_h, self.img_w = img_h, img_w
        self.processed_img_depth = processed_img_depth

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.processed_img_depth, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(257, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 2, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2)
        )

        self.masked_loss = MaskedLoss()

    @torch.cuda.amp.autocast()
    def forward(self, img_inputs, hue_inputs):
        # Encoder
        encoder_output = self.encoder(img_inputs)
        log(1, "encoder_output shape: {0}", encoder_output.shape)

        # Prepare hue input
        batch_size, _, h, w = encoder_output.shape
        hue_feature = hue_inputs.view(batch_size, 1, 1, 1).expand(-1, -1, h, w)

        # Concatenate encoder output and hue feature
        encoder_output = torch.cat([encoder_output, hue_feature], dim=1)

        # Decoder
        output = self.decoder(encoder_output)

        return output

    def compile(self, optimizer='adam'):
        self.optimizer = optim.Adam(
            self.parameters(), lr=0.000001
        ) if optimizer == 'adam' else optim.SGD(
            self.parameters(), lr=0.000001
        )

    def fit(self, dataloader, epochs=10, verbose=2):
        self.train()
        device = next(self.parameters()).device
        scaler = torch.cuda.amp.GradScaler()
        ll = 2

        for epoch in range(epochs):
            running_loss = 0.0
            for batch_idx, ((processed_img, hue), true_img) in enumerate(dataloader):
                processed_img = processed_img.to(device, non_blocking=True)\
                    .permute(0, 3, 1, 2)\
                    .to(torch.float16)
                hue = hue.to(device, non_blocking=True)
                true_img = true_img.to(device, non_blocking=True)\
                    .permute(0, 3, 1, 2)\
                    .to(torch.float16)
                log(ll, "processed_img shape: {0}", processed_img.shape)
                log(ll, "hue shape: {0}", hue.shape)
                log(ll, "true_img shape: {0}", true_img.shape)

                #self.optimizer.zero_grad(set_to_none=True)
                outputs = self(processed_img, hue)
                mask = processed_img[:, 1:2, :, :]
                mask = (mask == 255).float()  # Convert to binary mask

                # Calculate loss using the custom loss function
                loss = self.masked_loss(outputs, true_img, mask)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if verbose == 2 and batch_idx % 10 == 9:
                    log(3, f'[{epoch + 1}, {batch_idx + 1:5d}] loss: {running_loss / 10:.3f}')
                    running_loss = 0.0

        torch.cuda.synchronize()

    def predict(self, processed_img, requested_hue):
        self.eval()
        device = next(self.parameters()).device
        scaler = torch.cuda.amp.GradScaler()
        with torch.no_grad():
            processed_img = torch.from_numpy(processed_img).unsqueeze(0)\
                    .to(device, non_blocking=True)\
                    .permute(0, 3, 1, 2)\
                    .to(torch.float16)

            requested_hue = torch.tensor([requested_hue])\
                    .to(device, non_blocking=True)\
                    .to(torch.float16)
            prediction = self(processed_img, requested_hue)
        return prediction.permute(0, 2, 3, 1).squeeze(0).cpu().numpy().astype(np.uint8)

    def load_model(self, model_path=model_path):
        self.load_state_dict(torch.load(model_path))

    def save_model(self, model_path=model_path, index=None):
        if index is None:
            indexed_model_path = model_path
        else:
            model_name = path.basename(model_path)
            splitted_name = path.splitext(model_name)
            indexed_model_path = os.path.join(
                os.path.dirname(model_path),
                splitted_name[0] + '.' + str(index) + splitted_name[1]
            )
        torch.save(self.state_dict(), indexed_model_path)

class HairColorizationDataset(Dataset):

    def __init__(
        self,
        image_dir,
        output_dir,
        data_dir,
        num_train_samples,
        img_h, img_w,
        preprocess_image_fn,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sample_expansion_rate = 3 # it's defined by the number of data points.
        self.data_list = glob.glob(path.abspath(data_dir) + f'/*.json')
        assert len(self.data_list) >= num_train_samples,\
                "length of data_list: {0}, num_train_samples: {1}".format(
            len(self.data_list), num_train_samples
        )
        self.output_dir = path.abspath(output_dir)
        self.image_dir, self.data_dir = path.abspath(image_dir), path.abspath(data_dir)
        self.num_train_samples = num_train_samples
        self.img_h, self.img_w = img_h, img_w
        self.preprocess_image = preprocess_image_fn


    def __len__(self):
        return self.num_train_samples * self.sample_expansion_rate

    def __getitem__(self, idx):
        data_idx = idx // self.sample_expansion_rate
        hue_idx = idx % self.sample_expansion_rate
        ll = 2

        json_path = self.data_list[data_idx]
        processed_img, true_img, data = self.preprocess_image(json_path)

        hues = []
        hues.append(data['mean_hue'])
        hues.append(data['median_hue'])
        hues.append(data['mode_hue'])
        # This is how we define the self.sample_expansion_rate
        assert(len(hues) == self.sample_expansion_rate)

        hue = hues[hue_idx]
        return (
            (processed_img, hue),
            true_img,
        )

class HairColorization:

    def __init__(self, model_path=model_path):
        self. model_path = model_path
        self.img_h, self.img_w = 240, 240
        self.processed_img_cache_suffix = '-processed-img-cache-1.npy'
        self.true_img_cache_suffix = '-true-img-cache-0.jpg'
        self.processed_img_depth = 9

    def train_loop(
            self, image_dir, output_dir, data_dir,
            num_train_samples, batch_size, model_path=model_path,
            init_index=1,
            **kwargs
        ):

        def preprocess_image(json_path):
            return self.preprocess_image_load_cache(
                json_path, data_dir, withCache=True,
            )

        dataset = HairColorizationDataset(
            image_dir, output_dir, data_dir, num_train_samples,
            self.img_h, self.img_w,
            preprocess_image,
            **kwargs
        )
        model = HairColorizationModel(self.img_h, self.img_w, self.processed_img_depth)
        model.to('cuda')
        model.compile()
        if path.exists(model_path):
            model.load_model(model_path)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4
        )
        model.fit(dataloader)
        model.save_model(model_path)

    def predict(self, image_path, requested_hue, segmentor, hair_sgment_model_path, output_dir):
        """
            Evaluate the mode to predict.
        """
        ll = 3
        segmentor.load_model(hair_sgment_model_path)
        processed_img, true_img_hsv, img_rgb, b_mask, window = self.preprocess_image_with_cache(
                image_path, output_dir, segmentor, withCache=False
        )
        del segmentor.model
        if window is None:
            raise "withCache is False so window should not be None!"

        model = HairColorizationModel(self.img_h, self.img_w, self.processed_img_depth)
        model.to('cuda')
        if path.exists(model_path):
            model.load_model(model_path)
        prediction = model.predict(processed_img, requested_hue)
        log(ll, "predicted shape: {0}", prediction.shape)
        prediction = prediction[0]
        postprediction = np.zeros((self.img_h, self.img_w, 3))
        postprediction[:,:,:2] = prediction
        postprediction[:,:,2] = true_img_hsv[:,:,2:]
        postprediction_hsv = postprediction.astype(np.uint8)
        log(ll, "postprediction shape: {0}", postprediction_hsv.shape)
        logImage(ll, output_dir, postprediction_hsv)
        return self.postprocess_image(
            img_rgb, postprediction_hsv, b_mask, window, output_dir
        )

    def postprocess_image(self, img_rgb, prediction_hsv, b_mask, window, output_dir):
        """
            Postprocess image.
        """
        log(3, "predicted image: {0}", prediction_hsv)
        logImage(3, output_dir, prediction_hsv)
        window_top_left_y, window_top_left_x,\
            window_bottom_right_y, window_bottom_right_x = window
        window_size = (
            window_bottom_right_y - window_top_left_y,
            window_bottom_right_x - window_top_left_x,
        )
        processed_img_hsv = cv.resize(prediction_hsv, window_size, interpolation= cv.INTER_LINEAR)
        processed_img_rgb = cv.cvtColor(processed_img_hsv, cv.COLOR_HSV2RGB)
        logImage(3, output_dir, processed_img_rgb, img_space=LOG_RGB_IMAGE)

        postprocess_img_rgb = img_rgb.copy()
        log(
            3, "shape of processed_img: {0}, postprocess_image: {1}, window: {2}",
            processed_img_rgb.shape, postprocess_img_rgb.shape, window
        )
        for y in range(window_top_left_y, window_bottom_right_y):
            for x in range(window_top_left_x, window_bottom_right_x):
                if b_mask[y][x].any():
                    postprocess_img_rgb[y][x] = processed_img_rgb[y - window_top_left_y][x - window_top_left_x]

        return img_rgb, postprocess_img_rgb

    def preprocess_image(self, img_rgb, gray_img, b_mask, label_img, output_dir):
        """
            Preprocess image.
        """
        assert self.img_h / self.img_w == img_rgb.shape[0] / img_rgb.shape[1],\
            "the requested width/height ratio must be the same as ratio in the original image."
        assert self.img_h < img_rgb.shape[0],\
            "the requested width/height must be smaller than the original image."

        window_top_left_y, window_top_left_x = img_rgb.shape[0], img_rgb.shape[1]
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
        log(ll, "img shape: {0}", img_rgb.shape)
        log(ll, "expected_image_size_ratio: {0}", expected_image_size_ratio)
        log(ll, "before window_size: {0}", window_size)
        log(ll, "before window_center: {0}", window_center)
        if window_size[0] > window_size[1]:
            new_half_size_1 = (window_size[0] / expected_image_size_ratio) / 2
            window_size[1] = window_size[0] / expected_image_size_ratio
            if window_center[1] + new_half_size_1 >= img_rgb.shape[1]:
                window_center[1] = img_rgb.shape[1] - new_half_size_1 - 1
            elif window_center[1] - new_half_size_1 < 0:
                window_center[1] = new_half_size_1
        else:
            new_half_size_0 = (window_size[1] * expected_image_size_ratio) / 2
            window_size[0] = window_size[1] * expected_image_size_ratio
            if window_center[0] + new_half_size_0 >= img_rgb.shape[0]:
                window_center[0] = img_rgb.shape[0] - new_half_size_0 - 1
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


        b_mask = b_mask[window_top_left_y:window_bottom_right_y + 1,window_top_left_x: window_bottom_right_x + 1]
        b_mask = b_mask.reshape((b_mask.shape[0], b_mask.shape[1], 1))
        area = 0
        for y, m_y in enumerate(b_mask):
            for x, m in enumerate(m_y):
                if m:
                    area += 1

        masked_area = np.full(
            (b_mask.shape[0], b_mask.shape[1], 1),
            area, dtype=np.uint8
        )

        expected_img_rgb = img_rgb[window_top_left_y:window_bottom_right_y + 1,window_top_left_x: window_bottom_right_x + 1][:]
        gray_img = gray_img[window_top_left_y:window_bottom_right_y + 1,window_top_left_x: window_bottom_right_x + 1]
        gray_img = gray_img.reshape((gray_img.shape[0], gray_img.shape[1], 1))
        label_img = label_img[window_top_left_y:window_bottom_right_y + 1,window_top_left_x: window_bottom_right_x + 1][:]
        assert gray_img.shape[-1] + b_mask.shape[-1] + masked_area.shape[-1] + label_img.shape[-1] == self.processed_img_depth,\
            "depth of processed_img({0}) is not what expected, {1}!".format(
                gray_img.shape[-1]+ b_mask.shape[-1] + masked_area.shape[-1] + label_img.shape[-1],
                self.processed_img_depth
            )
        processed_img = np.full(
            (gray_img.shape[0], gray_img.shape[1], self.processed_img_depth),
            -1, dtype=np.uint8
        )
        processed_img[:,:,:1] = gray_img
        processed_img[:,:,1:2] = b_mask
        processed_img[:,:,2:3] = masked_area
        processed_img[:,:,3:] = label_img
        log(ll, "expected_img shape: {0}", expected_img_rgb.shape)
        log(ll, "processed_img shape: {0}", processed_img.shape)

        processed_img = cv.resize(processed_img, (self.img_h, self.img_w), interpolation= cv.INTER_LINEAR)
        expected_img_rgb = cv.resize(expected_img_rgb, (self.img_h, self.img_w), interpolation= cv.INTER_LINEAR)

        logImage(ll, output_dir, img_rgb, img_space=LOG_RGB_IMAGE)
        logImage(ll, output_dir, expected_img_rgb, img_space=LOG_RGB_IMAGE)

        log(ll, "processed_img shape: {0}", processed_img.shape)
        log(ll, "expected_img shape: {0}", expected_img_rgb.shape)
        window = (
            window_top_left_y,
            window_top_left_x,
            window_bottom_right_y,
            window_bottom_right_x,
        )

        return processed_img, expected_img_rgb, window

    def preprocess_dataset(self, input_dir, output_dir, hair_sgment_model_path):
        mod = importlib.import_module("hair_segment")
        reload(mod)
        segmentor = mod.HairSegment()
        segmentor.load_model(hair_sgment_model_path)
        for image_path in glob.glob(os.path.abspath(input_dir) + '/*.jpg'):
            processed_img, true_img, _, _, _ = self.preprocess_image_with_cache(
                    image_path, output_dir, segmentor, withCache=True,
            )
            if processed_img is None:
                log(3, "processed_img is None! {0}", image_path)
                continue

    def preprocess_image_with_cache(
            self, image_path, data_dir, segmentor, withCache=True
        ):
        """
            Preprocess image and cache it.
            If withCache is False, then `window` is None.
        """
        log(3, "image_path: {0}", image_path)
        image_name = os.path.basename(image_path)
        splitted_name = os.path.splitext(image_name)
        data_name = splitted_name[0] + '.json'
        data_output_path = os.path.join(data_dir, data_name)
        b_mask_name = splitted_name[0] + '-b-mask.npy'
        b_mask_output_path = os.path.join(data_dir, b_mask_name)
        label_img_name = splitted_name[0] + '-label-img.npy'
        label_img_output_path = os.path.join(data_dir, label_img_name)
        log(3, "data_output_path: {0}", data_output_path)
        if withCache and os.path.exists(data_output_path) \
            and os.path.exists(b_mask_output_path) and os.path.exists(label_img_output_path):
            with open(data_output_path, 'r') as f:
                try:
                    data = json.loads(f.read())
                except json.JSONDecodeError as e:
                    raise "cannot parse: {0}, {1}".format(data_output_path, e)
            img_rgb = cv.imread(image_path)
            if img_rgb is None:
                raise "cannot parse: {0}".format(img_path)
            img_rgb = cv.cvtColor(img_rgb, cv.COLOR_BGR2RGB)
            b_mask = np.load(b_mask_output_path)
            label_img = np.load(label_img_output_path)
        else:
            img_rgb, b_mask, data, label_img = segmentor.make_data(image_path)
            if data is None:
                return None, None, None, None
            if withCache:
                with open(data_output_path, 'w') as f:
                    json.dump(data, f)
                np.save(b_mask_output_path, b_mask)
                np.save(label_img_output_path, label_img)

        gray_img = cv.cvtColor(img_rgb, cv.COLOR_RGB2GRAY)

        processed_img_name = splitted_name[0] + self.processed_img_cache_suffix
        processed_img_path = os.path.join(data_dir, processed_img_name)
        true_img_name = splitted_name[0] + self.true_img_cache_suffix
        true_img_path = os.path.join(data_dir, true_img_name)

        if withCache and os.path.exists(processed_img_path) and os.path.exists(true_img_path):
            window = None
            processed_img = np.load(processed_img_path)
            true_img_hsv = cv.imread(true_img_path)
            if true_img_hsv is None:
                raise "cannot parse: {0}".format(true_img_path)
            true_img_bgr = cv.cvtColor(true_img_hsv, cv.COLOR_BGR2RGB)
        else:
            processed_img, true_img_bgr, window = self.preprocess_image(
                img_rgb, gray_img, b_mask, label_img, data_dir
            )
            if withCache:
                np.save(processed_img_path, processed_img)
                cv.imwrite(true_img_path, cv.cvtColor(true_img_hsv, cv.COLOR_RGB2BGR))
        true_img_hsv = cv.cvtColor(true_img_bgr, cv.COLOR_BGR2HSV)
        true_img_hsv = true_img_hsv[:,:,:2]
        return processed_img, true_img_hsv, img_rgb, b_mask, window


    def preprocess_image_load_cache(
            self, data_output_path, data_dir, withCache=True
        ):
        """
            Load preprocess image from cache.
        """
        data_name = os.path.basename(data_output_path)
        splitted_name = os.path.splitext(data_name)
        processed_img_name = splitted_name[0] + self.processed_img_cache_suffix
        processed_img_path = os.path.join(data_dir, processed_img_name)
        true_img_name = splitted_name[0] + self.true_img_cache_suffix
        true_img_path = os.path.join(data_dir, true_img_name)

        with open(data_output_path, 'r') as f:
            try:
                data = json.loads(f.read())
            except json.JSONDecodeError as e:
                raise "cannot parse: {0}, {1}".format(data_output_path, e)
        processed_img = np.load(processed_img_path)
        true_img_hsv = cv.imread(true_img_path)
        if true_img_hsv is None:
            raise "cannot parse: {0}".format(true_img_path)
        true_img_hsv = cv.cvtColor(true_img_hsv, cv.COLOR_BGR2HSV)
        true_img_hsv = true_img_hsv[:,:,:2]
        return processed_img, true_img_hsv, data


# Helper function to both segment the hairs and color them.
def predict(image_path, hue, hair_sgment_model_path, hair_colorization_model_path, output_dir):
    import gc
    import importlib
    import sys
    import torch
    torch.cuda.empty_cache()
    mod = importlib.import_module("hair_segment")
    reload(mod)
    segmentor = mod.HairSegment()
    colorizor = HairColorization(hair_colorization_model_path)
    img_rgb, result_rgb = colorizor.predict(image_path, hue, segmentor, hair_sgment_model_path, output_dir)
    del colorizor
    torch.cuda.empty_cache()

    logImage(3, output_dir, img_rgb, img_space=LOG_RGB_IMAGE)
    logImage(3, output_dir, result_rgb, img_space=LOG_RGB_IMAGE)
    return result_rgb


"""
Image spaces used by logger.
"""
LOG_HSV_IMAGE = 0
LOG_BGR_IMAGE = 1
LOG_RGB_IMAGE = 2

def logImage(level, output_dir, img, img_space=LOG_HSV_IMAGE):
    """
        Log image in Jupyter lab.
    """
    if level > 2:
       from IPython.display import display, Image

       img_path = output_dir + f'/.log_img.png'
       if img_space == LOG_HSV_IMAGE:
           img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
       elif img_space == LOG_RGB_IMAGE:
           img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
       cv.imwrite(img_path, img)
       display(Image(filename=img_path, height=600))

def logImageWithPath(level, output_dir, img_path):
    log(level, "file exists: {0}", os.path.exists(img_path))
    img = cv.imread(img_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    logImage(level, output_dir, img)

def log(level, s, *arg):
    if level > 2:
       if arg:
           print(s.format(*arg))
       else:
           print(s)


