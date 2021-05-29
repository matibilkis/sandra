import tensorflow as tf
import numpy as np
from networks import Encoder, Decoder
from sindy import SINDy

encoder = Encoder()
decoder = Decoder()

encoder(tf.random.uniform((1,128)))
decoder(tf.random.uniform((1,3)))


x = tf.random.uniform((1,128))
x_dot = tf.random.uniform((1,128))

encoder = Encoder()
encoder(x)

sindy =SINDy()
sindy(np.random.randn(3),0)
