from os.path import join
from typing import List

import torch.utils.data as data



class AbstractDataset(data.Dataset):

    def __init__(self):
        super().__init__()
        self.visits: dict
        self.path: str
        self.patches_from_single_image: int
        self.real_length: int
        self.multiplier: int
        self.record: dict

    def _load(self, _index):
        raise NotImplementedError

    def _make_dataset(self, patients: List) -> list:
        dataset = []

        for k in patients:
            # print(self.visits[k])
            for visit in self.visits[k]:
                # print(visit)
                record = {}
                record['path'] = join(
                    self.path,
                    k
                    # +'_'+visit['Position']+'.mat'
                    # str(visit['DayInStudy'])
                )
                record['FileSetId'] = visit['FileSetId']
                record['DayInStudy'] = visit['DayInStudy']
                record['PatId'] = k
                record['Position'] = visit['Position']

                dataset.append(record)
        # print(dataset)
        return dataset

    def __getitem__(self, index):
        index = index % self.real_length

        if self.patches_from_current_image >= self.patches_from_single_image:
            self._load(index)
            self.patches_from_current_image = 0

        self.patches_from_current_image += 1

        record = self.record.copy()

        if self.transforms is not None:
            record = self.transforms(record)

        return record

    def __len__(self):
        return int(self.multiplier * self.real_length)
