"""Implements a class to make time series dataset for training, testing, and validation."""  

import numpy as np
import pandas as pd
import tensorflow as tf

class WindowData(object):
    """
    Initializes a class for preprocessing and shaping data for working with time series.   
    
    It should be able to:  
        -  
    
    """   

    def __init__(self, input_width, label_width, shift, 
                train_ds, val_ds, test_ds 
                ):
        """
        Initializes the class.  

        Args:  
            - input_width: The number of time steps to use as input. 
                (e.g. the positions from 0 to 10 seconds)   

            - label_width: The number of time steps to use as label.  
                (e.g. the future positions from 10 to 20 seconds)     

            - shift: The number of time steps to shift the label.  
                (e.g. if we want to predict the 20 second position, the shift
                must be greater than 10 given the input_width)  

            - train_ds: The training dataset.

            - val_ds: The validation dataset.

            - test_ds: The test dataset.
        """
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.test_ds = test_ds  

        # window parameters  
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(0, input_width) # slice object from 0 to input_width
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None) # slice object from label_start to the end
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]


    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label start: {self.label_start}',
            f'Input slice: {self.input_slice}',
            f'Label slice: {self.labels_slice}',
        ])
    

    def split(self, arr):
        """
        Splits the array in windows.  

        Args:
            - arr: ndarray, the array to be split into windows.

        Output:
            - pair: tuple of (input, label) windows.
        
        """
        inputs = arr[:, self.input_slice, :]
        labels = arr[:, self.labels_slice, :]

        inputs.set_shape([None, self.input_width, None])  
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data, batch_size, shuffle=False):
        """
        Makes the dataset.  

        Args:
            - data: ndarray, the array to be split into windows.
            - batch_size: int, the batch size.
            - shuffle: bool, whether to shuffle the dataset or not, default is False.

        Output:
            - dataset: tf.data.Dataset, the dataset.
        
        """  
        assert isinstance(data, np.ndarray), 'data must be a numpy array'

        dataset = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=batch_size,
        )

        dataset = dataset.map(self.split)

        return dataset

    @property
    def make_train(self, batch_size=32, shuffle=False):
        return self.make_dataset(self.train_ds, batch_size, shuffle)

    @property
    def make_val(self, batch_size=32, shuffle=False):
        return self.make_dataset(self.val_ds, batch_size, shuffle)

    @property
    def make_test(self, batch_size=32, shuffle=False):
        return self.make_dataset(self.test_ds, batch_size, shuffle)

    