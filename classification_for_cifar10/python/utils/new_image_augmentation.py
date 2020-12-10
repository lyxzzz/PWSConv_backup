import cv2
import numpy as np
import types
from numpy import random

class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels=None):
        for t in self.transforms:
            img, labels = t(img, labels)
        return img, labels

class ReadImage(object):
    def __call__(self, image, labels=None):
        img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, labels

class ConvertFromInts(object):
    def __call__(self, image, labels=None):
        return image.astype(np.float32), labels

class Expand(object):
    def __init__(self, mean):
        self.mean = mean
    def __call__(self, image, labels):
        if random.randint(2):
            return image, labels

        height, width, depth = image.shape
        ratio = 4
        mean = self.mean
        left = random.uniform(0, width + 2 * ratio - width)
        top = random.uniform(0, height + 2 * ratio - height)

        expand_image = np.zeros(
            (int(height + 2 * ratio), int(width + 2 * ratio), depth),
            dtype=image.dtype)
        expand_image[:, :, :] = mean
        expand_image[int(top):int(top + height),
                     int(left):int(left + width)] = image
        image = expand_image[ratio:ratio+height, ratio:ratio+width, :]

        return image, labels

class RandomSaturation(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    def __call__(self, image, labels=None):
        if random.randint(2):
            image[:, :, 1] *= random.uniform(self.lower, self.upper)

        return image, labels


class RandomHue(object):
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, image, labels=None):
        if random.randint(2):
            image[:, :, 0] += random.uniform(-self.delta, self.delta)
            image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
        return image, labels


class RandomLightingNoise(object):
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]
            shuffle = SwapChannels(swap)  # shuffle channels
            image = shuffle(image)
        return image, labels


class ConvertColor(object):
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform
        self.current = current

    def __call__(self, image, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return image, labels


class RandomContrast(object):
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, image, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            mean = np.mean(image, axis=(0, 1))
            image = (image - mean) * alpha + mean
            image = np.clip(image, 0.0, 255.0)
        return image, labels


class RandomBrightness(object):
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, image, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            image += delta
            image = np.clip(image, 0.0, 255.0)
        return image, labels

class RandomMirror(object):
    def __call__(self, image, labels):
        if random.randint(2):
            image = image[:, ::-1]
        return image, labels

class CropResize(object):
    def __init__(self, crop_size=[256, 480], patch_size=224):
        self.crop_size = crop_size
        self.patch_size = patch_size

    def __call__(self, image, labels):
        cropsize = random.randint(self.crop_size[0], self.crop_size[1])
        h, w = image.shape[0:2]
        if h > w:
            h = int(h * cropsize / w)
            w = int(cropsize)
            
            set_h = random.randint(0, h - self.patch_size)
            set_w = random.randint(0, w - self.patch_size)
        else:
            w = int(w * cropsize / h)
            h = int(cropsize)

            set_h = random.randint(0, h - self.patch_size)
            set_w = random.randint(0, w - self.patch_size)

        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        patch = image[set_h:set_h + self.patch_size, set_w: set_w + self.patch_size, :]

        return patch, labels

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, labels):
        im = image.copy()
        im, abels = self.rand_brightness(im, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, labels = distort(im, labels)

        return im, labels

class Normalize(object):
    def __init__(self, mean=[125., 122., 113.], var=[51., 52., 51.]):
        self.mean = mean
        self.var = var
    def __call__(self, image, labels):
        image = (image - self.mean) / self.var

        return image, labels

class CalMeanAndVar(object):
    def __call__(self, image, labels=None):
        cropsize = 480
        h, w = image.shape[0:2]
        if h > w:
            h = int(h * cropsize / w)
            w = int(cropsize)
        else:
            h = int(h * cropsize / w)
            w = int(cropsize)
        
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        image = image.reshape([-1, image.shape[-1]])
        mean = np.mean(image, axis=0)
        var = np.var(image, axis=0)
        return mean, var

class Augmentation(object):
    def __init__(self, augment_dict):
        mean = augment_dict['dataset_distribution']['mean']
        var = augment_dict['dataset_distribution']['var']
        para = augment_dict['parameters']
        if augment_dict['type'] == "ImageNet":
            self.augment = Compose([
                ReadImage(),
                ConvertFromInts(),
                CropResize(para.get("crop_size", [256, 480]), para.get("patch_size", 224)),
                RandomMirror(),
                PhotometricDistort(),
                Normalize(mean, var),
            ])
        elif augment_dict['type'] == "cifar10":
            self.augment = Compose([
                ConvertFromInts(),
                RandomMirror(),
                Expand(mean),
                Normalize(mean, var),
            ])
        elif augment_dict['type'] == "cal_distribution":
            self.augment = Compose([
                ReadImage(),
                CalMeanAndVar(),
            ])
    def __call__(self, img, labels=None):
        return self.augment(img, labels)