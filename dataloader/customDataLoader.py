import torch.utils.data
from dataloader.base_data_loader import BaseDataLoader
from dataset.creatDataset import CreateDataset

class CustomDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)

        self.dataset = CreateDataset(opt)

        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=opt.random_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batchSize >= self.opt.max_dataset_size:
                break
            yield data
