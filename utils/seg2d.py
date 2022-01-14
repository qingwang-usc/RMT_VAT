import math
import numbers
import random

from PIL import Image, ImageOps
import numpy as np
import torch
import cv2
import torch.nn.functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        assert img.shape[0] == mask.shape[0]
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            return img.resize((tw, th), Image.BILINEAR), mask.resize((tw, th), Image.NEAREST)

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):

        img = Image.fromarray(img.astype('uint8'), 'RGB')
        mask = Image.fromarray(mask.astype('uint8'))

        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th)), mask.crop((x1, y1, x1 + tw, y1 + th))


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST)


class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return img.resize((ow, oh), Image.BILINEAR), mask.resize((ow, oh), Image.NEAREST)


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.shape[0] == mask.shape[0]
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (img.size == (w, h))

                return img.resize((self.size, self.size), Image.BILINEAR), mask.resize((self.size, self.size),
                                                                                       Image.NEAREST)

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return img.rotate(rotate_degree, Image.BILINEAR), mask.rotate(rotate_degree, Image.NEAREST)


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask):
        assert img.size == mask.size

        w = int(random.uniform(0.5, 2) * img.size[0])
        h = int(random.uniform(0.5, 2) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)

        return self.crop(*self.scale(img, mask))


class SlidingCropOld(object):
    def __init__(self, crop_size, stride_rate, ignore_label):
        self.crop_size = crop_size
        self.stride_rate = stride_rate
        self.ignore_label = ignore_label

    def _pad(self, img, mask):
        h, w = img.shape[: 2]
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'constant', constant_values=self.ignore_label)
        return img, mask

    def __call__(self, img, mask):
        assert img.size == mask.size

        w, h = img.size
        long_size = max(h, w)

        img = np.array(img)
        mask = np.array(mask)

        if long_size > self.crop_size:
            stride = int(math.ceil(self.crop_size * self.stride_rate))
            h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1
            w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1
            img_sublist, mask_sublist = [], []
            for yy in range(h_step_num):
                for xx in range(w_step_num):
                    sy, sx = yy * stride, xx * stride
                    ey, ex = sy + self.crop_size, sx + self.crop_size
                    img_sub = img[sy: ey, sx: ex, :]
                    mask_sub = mask[sy: ey, sx: ex]
                    img_sub, mask_sub = self._pad(img_sub, mask_sub)
                    img_sublist.append(Image.fromarray(img_sub.astype(np.uint8)).convert('RGB'))
                    mask_sublist.append(Image.fromarray(mask_sub.astype(np.uint8)).convert('P'))
            return img_sublist, mask_sublist
        else:
            img, mask = self._pad(img, mask)
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
            return img, mask


class SlidingCrop(object):
    def __init__(self, crop_size, stride_rate, ignore_label):
        self.crop_size = crop_size
        self.stride_rate = stride_rate
        self.ignore_label = ignore_label

    def _pad(self, img, mask):
        h, w = img.shape[: 2]
        pad_h = max(self.crop_size - h, 0)
        pad_w = max(self.crop_size - w, 0)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), 'constant')
        mask = np.pad(mask, ((0, pad_h), (0, pad_w)), 'constant', constant_values=self.ignore_label)
        return img, mask, h, w

    def __call__(self, img, mask):
        assert img.size == mask.size

        w, h = img.size
        long_size = max(h, w)

        img = np.array(img)
        mask = np.array(mask)

        if long_size > self.crop_size:
            stride = int(math.ceil(self.crop_size * self.stride_rate))
            h_step_num = int(math.ceil((h - self.crop_size) / float(stride))) + 1
            w_step_num = int(math.ceil((w - self.crop_size) / float(stride))) + 1
            img_slices, mask_slices, slices_info = [], [], []
            for yy in range(h_step_num):
                for xx in range(w_step_num):
                    sy, sx = yy * stride, xx * stride
                    ey, ex = sy + self.crop_size, sx + self.crop_size
                    img_sub = img[sy: ey, sx: ex, :]
                    mask_sub = mask[sy: ey, sx: ex]
                    img_sub, mask_sub, sub_h, sub_w = self._pad(img_sub, mask_sub)
                    img_slices.append(Image.fromarray(img_sub.astype(np.uint8)).convert('RGB'))
                    mask_slices.append(Image.fromarray(mask_sub.astype(np.uint8)).convert('P'))
                    slices_info.append([sy, ey, sx, ex, sub_h, sub_w])
            return img_slices, mask_slices, slices_info
        else:
            img, mask, sub_h, sub_w = self._pad(img, mask)
            img = Image.fromarray(img.astype(np.uint8)).convert('RGB')
            mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
            return [img], [mask], [[0, sub_h, 0, sub_w, sub_h, sub_w]]



class CustomToTensorAndNormalize(object):
    def __init__(self, img_mean, img_std=0.0):
        self.img_mean = img_mean
        self.img_std = img_std


    def __call__(self, img, mask):

        img = np.array(img)
        mask = np.array(mask)

        img, mean, std = [np.array(a, np.float32) for a in (img, self.img_mean, self.img_std)]
        img -= mean * 255
        img *= 1.0 / (255 * std)

        img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
        mask = torch.from_numpy(mask).float()

        if not _is_tensor_image(img):
            raise TypeError('tensor is not a torch images.')


        return img, mask

def _is_tensor_image(img):
    return torch.is_tensor(img)



class RandomRotationScale(object):

    """data transformation for skin lesion dataset, including 8 rotations and scaling.

    Returns:
        PIL Image: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b

    """
    def __init__(self, scale=[0.7, 1.1]):
        self.scale = np.random.uniform(scale[0], scale[1])

    def __call__(self, img, target):
        """
        Args:
            img (cv2 Image): Image to be transformed.
            target (cv2 Image): Image to be transformed.

        Returns:
            cv2 Image: Randomly transformed image.
            cv2 target: Randomly transformed target.
        """

        # input_size = 88
        # u - dep
        # input_size = 72


        input_size = 224
        #  randomly scale
        deps = int(input_size * self.scale)
        rows = int(input_size * self.scale)
        if deps % 2 != 0:
            deps = deps + 1
            rows = rows + 1

        # random crop
        random_x = np.random.randint(0, img.shape[0] - deps)
        random_y = np.random.randint(0, img.shape[1] - rows)
        cropp_img = img[random_x: random_x + deps, random_y: random_y + rows, :].copy()
        cropp_tumor = target[random_x: random_x + deps, random_y: random_y + rows].copy()

        flip_num = np.random.randint(0, 8)
        if flip_num == 1:
            cropp_img = np.flipud(cropp_img)
            cropp_tumor = np.flipud(cropp_tumor)
        elif flip_num == 2:
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)
        elif flip_num == 3:
            cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
            cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
        elif flip_num == 4:
            cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
            cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
        elif flip_num == 5:
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)
            cropp_img = np.rot90(cropp_img, k=1, axes=(1, 0))
            cropp_tumor = np.rot90(cropp_tumor, k=1, axes=(1, 0))
        elif flip_num == 6:
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)
            cropp_img = np.rot90(cropp_img, k=3, axes=(1, 0))
            cropp_tumor = np.rot90(cropp_tumor, k=3, axes=(1, 0))
        elif flip_num == 7:
            cropp_img = np.flipud(cropp_img)
            cropp_tumor = np.flipud(cropp_tumor)
            cropp_img = np.fliplr(cropp_img)
            cropp_tumor = np.fliplr(cropp_tumor)

        cropp_img = cv2.resize(cropp_img, (input_size, input_size), interpolation=cv2.INTER_CUBIC)
        cropp_tumor = cv2.resize(cropp_tumor, (input_size, input_size), interpolation=cv2.INTER_NEAREST)

        return cropp_img, cropp_tumor

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

#
# class ToTensor(object):
#     """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
#
#     Converts a PIL Image or numpy.ndarray (H x W x C) in the range
#     [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
#     if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
#     or if the numpy.ndarray has dtype = np.uint8
#
#     In the other cases, tensors are returned without scaling.
#     """
#
#     def __call__(self, pic, target):
#         """
#         Args:
#             pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
#
#         Returns:
#             Tensor: Converted image.
#         """
#         return F.to_tensor(pic), F.to_tensor(target)
#
#     def __repr__(self):
#         return self.__class__.__name__ + '()'

# class Normalize(object):
#     """Normalize a tensor image with mean and standard deviation.
#     Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
#     will normalize each channel of the input ``torch.*Tensor`` i.e.
#     ``input[channel] = (input[channel] - mean[channel]) / std[channel]``
#
#     .. note::
#         This transform acts out of place, i.e., it does not mutates the input tensor.
#
#     Args:
#         mean (sequence): Sequence of means for each channel.
#         std (sequence): Sequence of standard deviations for each channel.
#     """
#
#     def __init__(self, mean, std, inplace=False):
#         self.mean = mean
#         self.std = std
#         self.inplace = inplace
#
#     def __call__(self, tensor, target):
#         """
#         Args:
#             tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
#
#         Returns:
#             Tensor: Normalized Tensor image.
#         """
#         return F.normalize(tensor, self.mean, self.std, self.inplace), target
#
#     def __repr__(self):
#         return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


