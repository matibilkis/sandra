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


def compute_loss(models, data):
    x, x_dot = data
    encoder, decoder, sindy = models
    ### LOSS 0 and 1 ###
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(encoder.trainable_variables)
        tape.watch(decoder.trainable_variables)
        tape.watch(x)
        tape.watch(tf.convert_to_tensor(sindy.coeffs))

        z = encoder(x)
        tape.watch(z)
        x_quasi = decoder(z)

        loss0 = tf.keras.losses.MSE(x,x_quasi)
        zdot_sindy = sindy(tf.transpose(z))

        with tape.stop_recording():
            dpsi_dz = tf.squeeze(tape.jacobian(x_quasi, z))
        xdot_pred = tf.matmul(zdot_sindy,dpsi_dz, transpose_b=True)
        loss1 = tf.keras.losses.MSE(x_dot,xdot_pred)

        ### LOSS 2 and 4 ###
        with tape.stop_recording():
            dphi_dx = tf.squeeze(tape.jacobian(z,x))
        #tf.einsum('ij,j->i', dphi_dx, tf.squeeze(x_dot))
        zdot_pred = tf.matmul(x_dot, dphi_dx, transpose_b=True)
        encoder_output = np.squeeze(encoder(x))
        loss2 = tf.cast(tf.keras.losses.MSE(zdot_pred, zdot_sindy),tf.float32)

        with tape.stop_recording():
            dphi_dx = tf.squeeze(tape.jacobian(z,x))
        #tf.einsum('ij,j->i', dphi_dx, tf.squeeze(x_dot))
        zdot_pred = tf.matmul(x_dot, dphi_dx, transpose_b=True)
        loss3 = tf.cast(tf.keras.losses.MSE(zdot_pred, zdot_sindy),tf.float32)

        loss4 = tf.expand_dims(tf.einsum('ij->',tf.math.abs(sindy.coeffs)),axis=0)
        total_loss = loss0 + loss1 + loss2 + loss3 + loss4

    grads_sindy_coeffs = tape.gradient(total_loss, sindy.coeffs)
    grads_encoder = tape.gradient(total_loss, encoder.trainable_variables)
    grads_decoder = tape.gradient(total_loss, decoder.trainable_variables)

    gradients = [grads_encoder, grads_decoder, grads_sindy_coeffs]
    for model, gradient in zip(models, gradients):
        model.optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    losses = tf.stack([loss0,loss1,loss2,loss3,loss4])
    return total_loss, losses
