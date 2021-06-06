
import tensorflow as tf
from networks import Encoder, Decoder, MetaModel, TrainingCallback
import numpy as np
import ast
from sindy import SINDy
import os


x, x_dot = np.load("datalorenz/paper2048.npy").astype(np.float32)
x = tf.reshape(x,[-1,128])
x_dot = tf.reshape(x,[-1,128])
    
encoder = Encoder()
decoder = Decoder()
sindy = SINDy()
models = [encoder, decoder, sindy]
    
metamodel = MetaModel(models, total_epochs=int(1e4+1e3), when_zero_lambda3=1000)
metamodel.compile_models()
with tf.device("GPU:0"):
    history = metamodel.fit(x=x, y=x_dot, epochs=metamodel.total_epochs, batch_size=8000,
                    callbacks=[TrainingCallback(), tf.keras.callbacks.TensorBoard("logs/prueba3"),
                                  ])