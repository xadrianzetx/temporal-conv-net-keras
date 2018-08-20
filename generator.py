import numpy as np


class SeriesGenerator:
    def __init__(self, df, seq_length, split=0.8, log_transform=False):
        self.df = df
        self.seq_length = seq_length - 1
        self.transform = log_transform
        self.train_idx = 0
        self.train_lim = int(np.floor(len(self.df) * split))
        self.val_idx = self.train_lim + 1
        self.valid_lim = len(self.df)
        self._populate()

    def _populate(self):
        self.df.columns = ['x']
        if self.transform:
            self.df = self.df.apply(lambda x: np.log1p(x))
        self.df['y'] = self.df['x'].shift(-1)

    def batch(self):
        while True:
            x_train = self.df['x'].loc[self.train_idx:self.train_idx + self.seq_length].values.reshape(1, -1, 1)
            y_train = self.df['y'].loc[self.train_idx:self.train_idx + self.seq_length].values.reshape(1, -1, 1)
            x_val = self.df['x'].loc[self.val_idx:self.val_idx + self.seq_length].values.reshape(1, -1, 1)
            y_val = self.df['y'].loc[self.val_idx:self.val_idx + self.seq_length].values.reshape(1, -1, 1)
            self.train_idx += (self.seq_length + 1)
            self.val_idx += (self.seq_length + 1)
            self.train_idx = 0 if self.train_idx > self.train_lim else self.train_idx
            self.val_idx = self.train_lim + 1 if self.val_idx > self.valid_lim else self.val_idx
            yield x_train, y_train, x_val, y_val
