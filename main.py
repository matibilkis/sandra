import tensorflow as tf
from networks import Encoder, Decoder, MetaModel, TrainingCallback
import numpy as np
import ast
from sindy import SINDy
import os


x, x_dot = np.load("../datalorenz/paper2048.npy").astype(np.float32)
x = tf.reshape(x,[-1,128])
x_dot = tf.reshape(x,[-1,128])
    
    
for runni in range(10):
    if runni>0:
        indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
        idx = tf.random.shuffle(indices)
        x_data = tf.gather(x, idx)
        x_dot_data = tf.gather(x_dot, idx)
    else:
        x_data = x
        x_dot_data = x_dot
    encoder = Encoder()
    decoder = Decoder()
    sindy = SINDy()
    
    ## build (important for saving the model)
    encoder.build((None,128))
    decoder.build((None,3))
    
    models = [encoder, decoder, sindy]

    metamodel = MetaModel(models, total_epochs=11001, when_zero_lambda3=1000,namerun=runni)
    metamodel.compile_models()
    with tf.device("GPU:0"):
        history = metamodel.fit(x=x, y=x_dot, epochs=metamodel.total_epochs, batch_size=8000,
                        callbacks=[TrainingCallback(), tf.keras.callbacks.TensorBoard("/data/uab-giq/scratch/matias/sandra/logs/{}/".format(runni)),
                                      ])