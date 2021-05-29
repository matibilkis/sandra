import tensorflow as tf

"""
some notes:
(SI) which are applied at all15layers of the network, except for the last layer of the encoder and the last layer of the decoder
"""

class Encoder(tf.keras.Model):
    def __init__(self, seed_val=0.1):
        """
        """
        super(Encoder,self).__init__()
        self.l1 = tf.keras.layers.Dense(64,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
                    bias_initializer = tf.keras.initializers.Zeros())
        self.l2 = tf.keras.layers.Dense(32,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
                    bias_initializer = tf.keras.initializers.Zeros())
        self.loutput = tf.keras.layers.Dense(3,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
                    bias_initializer = tf.keras.initializers.Zeros())

    def call(self, inputs):
        f = tf.nn.sigmoid(self.l1(inputs))
        f = tf.nn.sigmoid(self.l2(f))
        f = self.loutput(f)
        return f



class Decoder(tf.keras.Model):
    def __init__(self, seed_val=0.1):
        """
        """
        super(Decoder,self).__init__()
        self.l1 = tf.keras.layers.Dense(32,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
                    bias_initializer = tf.keras.initializers.Zeros())
        self.l2 = tf.keras.layers.Dense(64,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
                    bias_initializer = tf.keras.initializers.Zeros())
        self.loutput = tf.keras.layers.Dense(128,kernel_initializer=tf.random_uniform_initializer(minval=-seed_val, maxval=seed_val),
                    bias_initializer = tf.keras.initializers.Zeros())

    def call(self, inputs):
        f = tf.nn.sigmoid(self.l1(inputs))
        f = tf.nn.sigmoid(self.l2(f))
        f = self.loutput(f)
        return f
