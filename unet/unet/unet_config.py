"""Configuration for U-Net training."""


class UNetConfig:
    """Config compatible with data_split.build_train_valid_datasets."""

    def __init__(self, data_root, crop_size=128, seed=42):
        class DatasetCfg:
            def __init__(self, root, crop):
                self.name = 'sen12mscr'
                self.root = root
                self.split = ['train', 'val', 'test']
                self.train_ratio = 0.8
                self.data_range = 1.0
                self.crop_size = crop

        self.dataset = DatasetCfg(data_root, crop_size)
        self.seed = seed
