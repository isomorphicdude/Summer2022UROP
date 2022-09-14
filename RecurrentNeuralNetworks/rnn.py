"""Implements only vanilla LSTM."""  

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential  

class embedder(tf.keras.layers.Layer):

    def __init__(self, input_dim, embedding_dim, batch_size, **kwargs):
        """
        Initializes the class.

        Args:  
            - input_dim: the input dimension, e.g. (10,21,2), sequence length = 10,
                number of objects = 21, number of features = 2.
            - embedding_dim: int, the embedding dimension, e.g. 64
            - batch_size: int, the batch size, e.g. 32
        """
        super(embedder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size

    def build(self, input_shape):
        self.embed_mat = self.add_weight(name='embedding_matrix',
                                         shape=(
                                                self.batch_size,
                                                self.input_dim[0], 
                                                self.input_dim[1],
                                                self.embedding_dim,
                                                self.input_dim[2]
                                                ),
                                         initializer='uniform',
                                         trainable=True)

        self.embed_bias = self.add_weight(name='embedding_bias',
                                            shape=(self.input_dim[1]*self.embedding_dim,),
                                            initializer='zeros',
                                            trainable=True)
        super(embedder, self).build(input_shape)


    def call(self, inputs):
        h = inputs[...,tf.newaxis]
        h = tf.einsum('zabcd,zabec->zabed', h, self.embed_mat)
        # h = tf.keras.layers.Reshape((-1, self.input_dim[0], self.input_dim[1]*self.embedding_dim))(h)
        h = tf.keras.layers.Reshape((self.input_dim[0], self.input_dim[1]*self.embedding_dim))(h)

        outputs = h + self.embed_bias
        return outputs