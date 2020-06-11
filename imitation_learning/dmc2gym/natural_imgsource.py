# Copyright (c) Facebook, Inc. and its affiliates.
import random
from collections import OrderedDict

import cv2
import numpy as np
import skvideo.io
import tqdm


class BackgroundMatting(object):
    """
    Produce a mask by masking the given color. This is a simple strategy
    but effective for many games.
    """

    def __init__(self, color):
        """
        Args:
            color: a (r, g, b) tuple or single value for grayscale
        """
        self._color = color

    def get_mask(self, img):
        return img == self._color


class ImageSource(object):
    """
    Source of natural images to be added to a simulated environment.
    """

    def get_image(self):
        """
        Returns:
            an RGB image of [h, w, 3] with a fixed shape.
        """
        pass

    def reset(self):
        """ Called when an episode ends. """
        pass


class FixedColorSource(ImageSource):
    def __init__(self, shape, color):
        """
        Args:
            shape: [h, w]
            color: a 3-tuple
        """
        self.arr = np.zeros((shape[0], shape[1], 3))
        self.arr[:, :] = color

    def get_image(self):
        return self.arr


class RandomColorSource(ImageSource):
    def __init__(self, shape):
        """
        Args:
            shape: [h, w]
        """
        self.shape = shape
        self.arr = None
        self._color = np.random.randint(0, 256, size=(3,))
        self.reset()

    def reset(self):
        self.arr = np.zeros((self.shape[0], self.shape[1], 3))
        self.arr[:, :] = self._color

    def get_image(self):
        return self.arr


class RandomColorStripSource(ImageSource):
    def __init__(self, shape, domain: str):
        """
        Args:
            shape: [h, w]
        """
        self.shape = shape
        self.num_bins = 5
        self.domain = domain
        self.bins, self.num_dims = self.get_bins_and_dims()
        self.arr = None
        self._color = np.random.randint(0, 256, size=(self.num_dims, self.num_bins, 3))
        self.reset()

    def reset(self):
        self.arr = {}

        offset = self.shape[0] // self.num_dims

        if self.domain == "walker":
            offset = 60 // self.num_dims

        possible_states = itertools.product(
            list(range(self.num_bins)), repeat=self.num_dims
        )

        for state in possible_states:

            color = np.zeros((self.shape[0], self.shape[1], 3))

            for dim_idx, bin_idx in enumerate(state):
                color[offset * dim_idx : offset * (dim_idx + 1), :] = self._color[
                    dim_idx, bin_idx
                ]

            self.arr[",".join([str(x) for x in state])] = color

    def get_image(self, internal_state: OrderedDict):
        hash = self.get_hash(internal_state)
        return self.arr[hash]

    def get_bins_and_dims(self) -> np.array:
        if self.domain == "finger":
            num_dims = 3
            if self.num_bins == 5:
                bins = np.asarray(
                    [
                        [-45.51913226, -16.58326687, 12.35259852, 41.28846391],
                        [0.0, 2.49450377, 4.98900753, 7.4835113],
                        [0.0, 2.49450377, 4.98900753, 7.4835113],
                    ]
                )
        elif self.domain == "cheetah":
            num_dims = 3
            if self.num_bins == 5:
                bins = np.asarray(
                    [
                        [-9.1726065, -4.44320088, 0.28620475, 5.01561038],
                        [-9.07190738, -2.62753692, 3.81683355, 10.26120401],
                        [-3.64552829, -1.88143167, -0.11733506, 1.64676155],
                    ]
                )
        elif self.domain == "walker":
            num_dims = 3
            if self.num_bins == 5:
                bins = np.asarray(
                    [
                        [-9.07777307, -4.87188815, -0.66600323, 3.5398817],
                        [-7.24673993, -2.43037186, 2.38599621, 7.20236428],
                        [-16.92765027, -10.10602417, -3.28439807, 3.53722803],
                    ]
                )
        return bins, num_dims

    def get_hash(self, internal_state: OrderedDict):
        if self.domain == "finger":
            state_vec = internal_state["velocity"]
            hash = []
            for idx, state in enumerate(state_vec):
                hash.append(np.searchsorted(self.bins[idx], state))
        elif self.domain == "cheetah":
            state_vec = internal_state["velocity"][:3]
            hash = []
            for idx, state in enumerate(state_vec):
                hash.append(np.searchsorted(self.bins[idx], state))
        elif self.domain == "walker":
            state_vec = internal_state["velocity"][:3]
            hash = []
            for idx, state in enumerate(state_vec):
                hash.append(np.searchsorted(self.bins[idx], state))
        return ",".join([str(x) for x in hash])


class NoiseSource(ImageSource):
    def __init__(self, shape, strength=255):
        """
        Args:
            shape: [h, w]
            strength (int): the strength of noise, in range [0, 255]
        """
        self.shape = shape
        self.strength = strength

    def get_image(self):
        return np.random.randn(self.shape[0], self.shape[1], 3) * self.strength


class RandomImageSource(ImageSource):
    def __init__(self, shape, filelist, total_frames=None, grayscale=False):
        """
        Args:
            shape: [h, w]
            filelist: a list of image files
        """
        self.grayscale = grayscale
        self.total_frames = total_frames
        self.shape = shape
        self.filelist = filelist
        self.build_arr()
        self.current_idx = 0
        self.reset()

    def build_arr(self):
        self.total_frames = (
            self.total_frames if self.total_frames else len(self.filelist)
        )
        self.arr = np.zeros(
            (self.total_frames, self.shape[0], self.shape[1])
            + ((3,) if not self.grayscale else (1,))
        )
        for i in range(self.total_frames):
            # if i % len(self.filelist) == 0: random.shuffle(self.filelist)
            fname = self.filelist[i % len(self.filelist)]
            if self.grayscale:
                im = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)[..., None]
            else:
                im = cv2.imread(fname, cv2.IMREAD_COLOR)
            self.arr[i] = cv2.resize(
                im, (self.shape[1], self.shape[0])
            )  ## THIS IS NOT A BUG! cv2 uses (width, height)

    def reset(self):
        self._loc = np.random.randint(0, self.total_frames)

    def get_image(self):
        return self.arr[self._loc]


class RandomVideoSource(ImageSource):
    def __init__(self, shape, filelist, total_frames=None, grayscale=False):
        """
        Args:
            shape: [h, w]
            filelist: a list of video files
        """
        self.grayscale = grayscale
        self.total_frames = total_frames
        self.shape = shape
        self.filelist = filelist
        self.build_arr()
        self.current_idx = 0
        self.reset()

    def build_arr(self):
        if not self.total_frames:
            self.total_frames = 0
            self.arr = None
            random.shuffle(self.filelist)
            for fname in tqdm.tqdm(
                self.filelist, desc="Loading videos for natural", position=0
            ):
                if self.grayscale:
                    frames = skvideo.io.vread(fname, outputdict={"-pix_fmt": "gray"})
                else:
                    frames = skvideo.io.vread(fname)
                local_arr = np.zeros(
                    (frames.shape[0], self.shape[0], self.shape[1])
                    + ((3,) if not self.grayscale else (1,))
                )
                for i in tqdm.tqdm(
                    range(frames.shape[0]), desc="video frames", position=1
                ):
                    local_arr[i] = cv2.resize(
                        frames[i], (self.shape[1], self.shape[0])
                    )  ## THIS IS NOT A BUG! cv2 uses (width, height)
                if self.arr is None:
                    self.arr = local_arr
                else:
                    self.arr = np.concatenate([self.arr, local_arr], 0)
                self.total_frames += local_arr.shape[0]
        else:
            self.arr = np.zeros(
                (self.total_frames, self.shape[0], self.shape[1])
                + ((3,) if not self.grayscale else (1,))
            )
            total_frame_i = 0
            file_i = 0
            with tqdm.tqdm(
                total=self.total_frames, desc="Loading videos for natural"
            ) as pbar:
                while total_frame_i < self.total_frames:
                    # if file_i % len(self.filelist) == 0: random.shuffle(self.filelist)
                    fname = self.filelist[file_i % len(self.filelist)]
                    if self.grayscale:
                        frames = skvideo.io.vread(
                            fname, outputdict={"-pix_fmt": "gray"}
                        )
                    else:
                        frames = skvideo.io.vread(fname)
                    for frame_i in range(frames.shape[0]):
                        if total_frame_i >= self.total_frames:
                            break
                        self.arr[total_frame_i] = cv2.resize(
                            frames[frame_i], (self.shape[1], self.shape[0])
                        )[
                            ..., None
                        ]  ## THIS IS NOT A BUG! cv2 uses (width, height)
                        pbar.update(1)
                        total_frame_i += 1

    def reset(self):
        self._loc = np.random.randint(0, self.total_frames)

    def get_image(self):
        img = self.arr[self._loc % self.total_frames]
        self._loc += 1
        return img
