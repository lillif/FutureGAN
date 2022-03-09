from datetime import datetime, time, timedelta
from functools import lru_cache
from random import randrange
from typing import Iterator, T_co

import numpy as np
import xarray as xr
from numpy import float32
import torch
from torch.utils.data import IterableDataset


class ClimateHackDataset(IterableDataset):
    """
    This is a basic dataset class to help you get started with the Climate Hack.AI
    dataset. You are heavily encouraged to customise this to suit your needs.

    Notably, you will most likely want to modify this if you aim to train
    with the whole dataset, rather than just a small subset thereof.

    ** customizations **
    Args:
        transform: Function to perform transformations on video frames (callable, optional)
    """

    def __init__(
        self,
        dataset: xr.Dataset,
        start_date: datetime = None, # first date to use
        end_date: datetime = None, # last date to use
        crops_per_slice: int = 1,
        day_limit: int = 0, # day limit - change for validation set
        transform = None, # * added for FutureGan compatibility *
    ) -> None:
        super().__init__()

        self.dataset = dataset
        self.crops_per_slice = crops_per_slice
        self.day_limit = day_limit
        self.cached_items = []

        times = self.dataset.get_index("time")
        self.min_date = times[0].date()
        self.max_date = times[-1].date()

        if start_date is not None:
            self.min_date = max(self.min_date, start_date)

        if end_date is not None:
            self.max_date = min(self.max_date, end_date)
        elif self.day_limit > 0:
            self.max_date = min(
                self.max_date, self.min_date + timedelta(days=self.day_limit)
            )
        self.transform = transform # * added for FutureGan compatibility *

    def _image_times(self, start_time, end_time):
        date = self.min_date
        while date <= self.max_date:
            current_time = datetime.combine(date, start_time)
            while current_time.time() <= end_time:
                yield current_time
                # ** only every 20 minutes **
                current_time += timedelta(minutes=20)

            date += timedelta(days=1)

    def _get_crop(self, input_slice, target_slice):
        '''
        Returns:
            satellite_sample:   torch.FloatTensor containing one satellite sample of nframes
                                with pixel range [-1,1]. (torch.FloatTensor)
                                Shape of torch.FloatTensor: [C,D,H,W].
                                (C: nimg_channels, D: nframes, H: img_h, W: img_w)
        '''

        # roughly over the mainland UK
        rand_x = randrange(550, 950 - 128)
        rand_y = randrange(375, 700 - 128)

        # make a data selection
        selection = input_slice.isel(
            x=slice(rand_x, rand_x + 128),
            y=slice(rand_y, rand_y + 128),
        )

        # get the OSGB coordinate data
        osgb_data = np.stack(
            [
                selection["x_osgb"].values.astype(float32),
                selection["y_osgb"].values.astype(float32),
            ]
        )

        if osgb_data.shape != (2, 128, 128):
            return None

        # ** changed for FutureGan to enable transformation to different size **

        # get the input satellite imagery
        input_data = selection["data"].values.astype(float32)
        if input_data.shape != (12, 128, 128):
            return None

        # get the target output
        # make a data selection
        target_selection = target_slice.isel(
            x=slice(rand_x, rand_x + 128),
            y=slice(rand_y, rand_y + 128),
        )

        target_data = target_selection["data"].values.astype(float32)
        if target_data.shape != (24, 128, 128):
            print(f"target data pre transform has shape {target_data.shape}!")
            return None

        # target_output = (
        #     target_slice["data"]
        #     .isel(
        #         x=slice(rand_x + 32, rand_x + 96),
        #         y=slice(rand_y + 32, rand_y + 96),
        #     )
        #     .values.astype(float32)
        # )

        # if target_output.shape != (24, 64, 64):
        #     return None

        # * change to be used by FutureGAN *

        # put input and output together
        input_torch = torch.from_numpy(input_data)
        target_torch = torch.from_numpy(target_data)
        satellite_data = torch.vstack((input_torch, target_torch))

        # put samples into list so FutureGan code can be used
        # adding dummy dimension in front as it expects channel dimension
        satellite_sample = [satellite_data[None,i, :, :] for i in range(36)]

        # transform images as needed
        if self.transform is not None:
            satellite_sample = [self.transform(image).mul(2).add(-1) for image in satellite_sample]
        
        # make torch.FloatTensor from satellite sample list
        # permute to get dimension order [C,D,H,W]
        satellite_sample = torch.stack(satellite_sample, 0).permute(1,0,2,3) 
        return satellite_sample

        # if self.transform is not None:
        #     input_data_frames = []
        #     for i in range(12):
        #         input_data_frames.append(input_data[i,:,:])
        #     input_data = [self.transform(frame).mul(2).add(-1) for frame in input_data_frames]
            
        #     target_data_frames = []
        #     for i in range(24):
        #         target_data_frames.append(target_data[i,:,:])
        #     target_data = [self.transform(frame).mul(2).add(-1) for frame in target_data_frames]
            
        #     # put both together for FutureGan
        #     video_data = input_data + target_data

        #     video_sample = torch.stack(video_data, 0).permute(1,0,2,3)
        #     ''' video sample: shape of torch.FloatTensor: [C,D,H,W].
        #                   (C: nimg_channels, D: nframes, H: img_h, W: img_w)
        #     '''
        #     # video_sample = [self.transform(frame).mul(2).add(-1) for frame in video_sample]

        # return osgb_data, input_data, target_data # * changed for FutureGan *
        # # return osgb_data, input_data, target_output

    def __iter__(self) -> Iterator[T_co]:
        if self.cached_items:
            for item in self.cached_items:
                yield item

            return
        
        # ** only three hour sequences starting from 9am to 2pm **
        # ** 16 per day **
        start_time = time(9, 0)
        end_time = time(14, 0)

        for current_time in self._image_times(start_time, end_time):
            data_slice = self.dataset.loc[
                {
                    "time": slice(
                        current_time,
                        current_time + timedelta(hours=2, minutes=55),
                    )
                }
            ]

            if data_slice.sizes["time"] != 36:
                continue

            input_slice = data_slice.isel(time=slice(0, 12))
            target_slice = data_slice.isel(time=slice(12, 36))

            crops = 0
            while crops < self.crops_per_slice:
                crop = self._get_crop(input_slice, target_slice)
                if crop != None:
                    self.cached_items.append(crop)
                    yield crop

                crops += 1
    
    def __len__(self):
        return len(self.dataset['data'])