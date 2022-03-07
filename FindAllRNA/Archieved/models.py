import os

import warnings

from encoders import AbstractSequenceEncoder
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
from tensorflow import keras
import numpy as np
# from tensorflow.python.keras.backend import GraphExecutionFunction

from tensorflow.keras.layers import LeakyReLU
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# workaround to solve keras __name__ issue
LR = LeakyReLU()
LR.__name__ = 'relu'


def buildCNNModel(inshape, num_classes, nlayers=2, cnndim=2):
    n = 32
    model = keras.Sequential()
    for i in range(nlayers):
        if (i==0):
            if (cnndim==2):
                model.add(keras.layers.Conv2D(n*(2**(i)),(3),padding='same',input_shape=inshape, activation=LR))
            else:
                model.add(keras.layers.Conv1D(n*(2**(i)),(3),padding='same',input_shape=inshape, activation=LR))
        else:
            if (cnndim==2):
                model.add(keras.layers.Conv2D(n*(2**(i)),(3),padding='same',activation=LR))
            else:
                model.add(keras.layers.Conv1D(n*(2**(i)),(3),padding='same',activation=LR))
        if (cnndim==2):
            model.add(keras.layers.MaxPooling2D(2))
        else:
            model.add(keras.layers.MaxPooling1D(2))
        model.add(keras.layers.Dropout(rate=0.25))
        
    if (nlayers>0):
        model.add(keras.layers.Flatten())
    else:
        model.add(keras.layers.Dense(500, input_shape=inshape, activation=LR))

    model.add(keras.layers.Dense(1000, activation=LR))
    model.add(keras.layers.Dense(100, activation=LR))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(num_classes, activation=tf.nn.softmax))
    return model

def buildCNNModelImproved1D(inshape,num_classes):

    model = keras.Sequential()
    
    model.add(keras.layers.Conv1D(128,10,padding='same',input_shape=inshape))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.5))
    model.add(keras.layers.MaxPooling1D(2))

    model.add(keras.layers.Conv1D(128,10,padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.5))
    model.add(keras.layers.MaxPooling1D(4))

    model.add(keras.layers.GaussianNoise(0.3))
    
    model.add(keras.layers.Conv1D(256,10,padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.5))
    model.add(keras.layers.MaxPooling1D(2))

    model.add(keras.layers.Conv1D(256,10,padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.5))
    model.add(keras.layers.MaxPooling1D(4))

    model.add(keras.layers.GaussianNoise(0.3))
    
    model.add(keras.layers.Conv1D(256,10,padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.5))
    model.add(keras.layers.MaxPooling1D(4))
    model.add(keras.layers.GaussianNoise(0.3))
    
    model.add(keras.layers.Dropout(rate=0.2))
    
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(128))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.5))
    
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.5))

    model.add(keras.layers.Dense(num_classes, activation=tf.nn.softmax))
    return model

def buildCNNModelImproved2D(inshape,num_classes):

    model = keras.Sequential()
    
    model.add(keras.layers.Conv2D(128,(3,3),padding='same',input_shape=inshape,
                                 kernel_regularizer=keras.regularizers.l2(0.003)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.5))

    model.add(keras.layers.Conv2D(128,(3,3),padding='same'))   
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.5))
    model.add(keras.layers.MaxPooling2D((2,2)))

    model.add(keras.layers.GaussianNoise(0.3))
    
    model.add(keras.layers.Conv2D(256,(3,3),padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.5))
    
    model.add(keras.layers.Conv2D(256,(3,3),padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.5))
    model.add(keras.layers.MaxPooling2D((2,2)))

    model.add(keras.layers.GaussianNoise(0.3))
    
    model.add(keras.layers.Conv2D(256,(3,3),padding='same'))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.5))
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.GaussianNoise(0.3))
    
    model.add(keras.layers.Dropout(rate=0.2))
    
    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(128))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.5))
    
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU(alpha=0.5))

    model.add(keras.layers.Dense(num_classes, activation=tf.nn.softmax))
    return model


class EntropyClassifier:
    def __init__(self, model: GraphExecutionFunction, encoder: AbstractSequenceEncoder, classes: list, threshold: float):
        
        assert isinstance(model, GraphExecutionFunction)
        assert isinstance(encoder, AbstractSequenceEncoder)
        assert isinstance(classes, list)
        assert isinstance(threshold, float)
        
        self.T = threshold
        self.model = model
        self.encoder = encoder
        self.classes = classes
        
        
    def _calculate_entropy(self, test_seqs, iterations):
        avrp_rnd = np.zeros((len(test_seqs), len(self.classes)))
        avrhp_rnd = np.zeros((len(test_seqs)))
        
        for _ in range(iterations):
            preds_rnd = self.model([test_seqs,1])
            avrp_rnd = avrp_rnd + preds_rnd[0]
            avrhp_rnd = avrhp_rnd + np.sum(-preds_rnd[0]*np.log2(preds_rnd[0]+1e-10),1)
            
        avrp_rnd = avrp_rnd/iterations
        avrhp_rnd = avrhp_rnd/iterations
        hp_rnd = np.sum(-avrp_rnd*np.log2(avrp_rnd+1e-10),1)
        
        return hp_rnd
    

    def predict(self, test_seqs):
        """
        Predicts if given RNA sequences are functional (non-coding) or non-function.
        Returns 1 for functional RNAs and 0 for non-functional.
        """
        entropies = self._calculate_entropy(test_seqs, iterations=50)
        return np.where(entropies > self.T, 0, 1)
    
    



if __name__ == '__main__':
    m = buildCNNModel(inshape=(200, 16), num_classes=3, nlayers=2, cnndim=1)
    print(m.summary())
