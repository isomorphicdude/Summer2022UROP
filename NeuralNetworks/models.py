import argparse
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1, l2, l1_l2

def getModel(input_shape = (7,),
            num_layers   = 2,
            hidden_units = [14,7],
            output_shape = (1,),
            activation = 'elu',
            initializer = tf.random_normal_initializer(mean=0.0, stddev=0.1),
            regularizer = l1_l2(0.000001,0.000001),
            final_activation = None,
            dropout = None,
            batchnorm = False
            ): 
    """
    Returns a fully-connected neural network model for training and testing.  

    Args:
        - input_shape: shape of the input data
        - num_layers: int, number of hidden layers
        - hidden_units: list of number of hidden units in each layer
        - output_shape: shape of the output data
        - activation: string, activation function
        - initializer: initializer for the weights
        - regularizer: regularizer for the weights
        - final_activation: string, activation function of final layer
        - dropout: list, dropout rate for each layer, default None
        - batchnorm: bool, specifies if batch normalization is used, default False 
    
    Output:  
        - model: tf.keras.Model, compiled if compile is True
    """  
    assert num_layers == len(hidden_units), "Number of hidden units must match number of layers"
    if dropout is not None:  
        assert num_layers == len(dropout), "Number of dropout rates must match number of layers"

    inputs = Input(shape=input_shape)
    h = Flatten()(inputs)

    for i, layer in enumerate(hidden_units):
        h = Dense(layer, activation=activation, 
                                  kernel_initializer = initializer,
                                  kernel_regularizer = regularizer)(h)
        if dropout:
            h = Dropout(dropout[i])(h)
        if batchnorm:
            h = BatchNormalization()(h)
    if final_activation is not None:
        outputs = Dense(output_shape[0], activation=final_activation,
                                        kernel_initializer = initializer)(h)
    else:
        outputs = Dense(output_shape[0], 
                                        kernel_initializer = initializer)(h)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)  

    return model   