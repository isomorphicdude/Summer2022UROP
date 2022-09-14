"""Implements the custom tuner class."""  

import keras_tuner
import numpy as np
import tensorflow as tf 

from models.models import getModel


##################################################################################################
def tuned_model(hp):
    """
    Returns a compiled hyperModel for keras tuner.  

    - Number of layers: 1-5
    - Number of hidden units: 20 - 50, step 5
    - Learning rate: 1e-4 - 1e-2, log sampling
    - Rate of lr decay: 0.85-0.9995
    - l1_coeff: 1e-8 - 1e-6.5, log sampling
    - l2_coeff: 1e-8 - 1e-6.5, log sampling
    - Loss: 
    - Metrics:
    """  

    # defining a set of hyperparameters for tuning and a range of values for each
    num_layers = hp.Int('num_layers', min_value=1, max_value=5) 

    # activation = hp.Choice('activation', ['elu', 'tanh'])

    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=0.01, sampling = 'log')
    rate_decay = hp.Float('rate_decay', min_value=0.85, max_value=0.9995)
    l1_reg = hp.Float('l1_coeff', min_value=10**(-8), max_value=10**(-6.5))
    l2_reg = hp.Float('l2_coeff', min_value=10**(-8), max_value=10**(-6.5))
    
    
    hidden_units = []
    for i in range(num_layers):
        hidden_unit = hp.Int(f'units_{i+1}', min_value=20, max_value=50, step=5)
        hidden_units.append(hidden_unit)

    model = getModel(input_shape=input_shape_glob,
                    output_shape=output_shape_glob,
                    num_layers = num_layers, 
                    hidden_units = hidden_units,
                    activation = 'elu',
                    regularizer = tf.keras.regularizers.l1_l2(l1_reg,l2_reg)
                    )

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        learning_rate, decay_steps = 4000, decay_rate = rate_decay, staircase = True)
    
    # perhaps a little change here with loss and metrics
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule), loss = tf.keras.losses.MeanAbsolutePercentageError(), 
                metrics = [tf.keras.metrics.MeanSquaredError()])

    return model


##################################################################################################


############################################################################################################
# for the moment being, we only use the RandomSearch tuner, 
# as other tuners need more thorough explanation
# Namely, we need to describe how BayesianOptimization and Hyperband work under the hood

class customTuner(keras_tuner.RandomSearch):

    def __init__(self, input_shape, output_shape, dim=None, basket=False, **kwargs):
        """
        Initializes the custom tuner class.    
        
        Args:  
            - input_shape: the shape of the input data
            - output_shape: the shape of the output data
        """  
        global input_shape_glob
        global output_shape_glob

        super(customTuner, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.output_shape = output_shape

        input_shape_glob = input_shape
        output_shape_glob = output_shape


    def run_trial(self, trial, train_ds, valid_ds, epochs, **kwargs):
        # overrides the run_trial method of the RandomSearch class
        # should return the result of model.fit()
        hp = trial.hyperparameters  
    
        compiled_model = tuned_model(hp)
        history = compiled_model.fit(train_ds, validation_data=valid_ds, epochs=epochs, **kwargs)
        return  history



# define a hypermodel subclass

class customHyperModel(keras_tuner.HyperModel):

  def build(self, hp):
    return tuned_model(hp)