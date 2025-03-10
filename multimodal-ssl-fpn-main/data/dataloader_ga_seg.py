from os.path import join
from typing import Dict, List
import json

import numpy as np
from skimage import io
import scipy.io
from PIL import Image 
import matplotlib.pyplot as plt
import io
import os
from os.path import join
import shutil  # For removing directories

from data.abstract_dataloader import AbstractDataset



class MultimodalGASegDataset(AbstractDataset):
    def __init__(
        self,
        paths: Dict,
        patients: List,
        multiplier=1,
        patches_from_single_image=1,
        transforms=None,
        get_spacing=False,
        reconstruction='None',
    ):
        super().__init__()
        self.path = paths['data']
        self.multiplier = multiplier
        self.patches_from_single_image = patches_from_single_image
        self.transforms = transforms
        self.get_spacing = get_spacing
        self.patients = patients
        self.reconstruction = reconstruction
        with open(paths['info'], 'r') as fp:
            self.visits = json.load(fp)

        self.dataset = self._make_dataset(patients=self.patients)

        self.real_length = len(self.dataset)
        print('# of scans:', self.real_length)

        self.patches_from_current_image = self.patches_from_single_image

    def _load(self, index):
        self.record = self.dataset[index].copy()
        path = self.record['path']
        file_set_id = self.record['FileSetId']

        mat = scipy.io.loadmat(path + "\\" + file_set_id + '_l.mat')
        image = mat["d3"]

        image_transposed = np.transpose(image, (1, 0, 2))

        image_transposed = image_transposed[None]
        self.record['image'] = image_transposed

        # if self.get_spacing:
        #     self.record['spacing'] = np.load(
        #         join(path, 'spacing.'+file_set_id+'.npy')
        #     )

        #     mat = scipy.io.loadmat(path + "\\" + file_set_id + '_l.mat')
        #     self.record['spacing'] = mat["d3"]

        prefix = 'preprocessed_images/bscan_size.'

        if self.reconstruction == 'slo':
            mask = io.imread(
                join(
                    path,
                    prefix+'slo.'+file_set_id+'.png'
                )
            ) # type: np.ndarray
            mask = mask/256
        elif self.reconstruction.endswith('faf'):
 
            mat = scipy.io.loadmat(path + "\\" + file_set_id + '_l.mat')
            array_mask = mat["d3"]

            mask = np.mean(array_mask, axis = 1)
            mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))

            image_buffer = io.BytesIO()
            plt.imshow(mask, cmap='gray', aspect='auto')
            plt.axis('off')
            plt.savefig(image_buffer, format="PNG", bbox_inches='tight', pad_inches=0)
            plt.close()
            image_bytes = image_buffer.getvalue()
            
            mask = mask/256
            if self.reconstruction.startswith('inverted'):
                mask = 1 - mask
        else:
            mat = scipy.io.loadmat(path + "\\" + file_set_id + '_l.mat')
            array_mask = mat["d3"]

            array_mask = np.transpose(array_mask, (1, 0, 2))

            mask = np.mean(array_mask, axis = 1)
            mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))

            image_buffer = io.BytesIO()
            plt.imshow(mask, cmap='gray', aspect='auto')
            plt.axis('off')
            plt.savefig(image_buffer, format="PNG", bbox_inches='tight', pad_inches=0)
            plt.close()
            image_bytes = image_buffer.getvalue()
            
            # mask = mask/256
            # Apply threshold
            # mask = np.where(mask>=0.5, 1., 0.)
        self.record['mask'] = mask[None,:,None,:]

        save_dir = "en_face_images"
        os.makedirs(save_dir, exist_ok=True)

        save_path = join(save_dir, f"{file_set_id}_en_face_image.png")

        en_face_image = Image.open(io.BytesIO(image_bytes))
        en_face_image.save(save_path)
        en_face_image.close()
        del image_bytes

        # return self.record