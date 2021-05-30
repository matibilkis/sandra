import tensorflow as tf

"""
some notes:
(SI) which are applied at all15layers of the network, except for the last layer of the encoder and the last layer of the decoder
"""



class MetaModel(tf.keras.Model):
    def __init__(self, models):
        super(MetaModel, self).__init__()
        self.encoder, self.decoder, self.sindy = models
        self.compile_models()
        self.total_loss = Metrica(name="energy")
        self.loss0 = Metrica(name="Loss_0")
        self.loss1 = Metrica(name="Loss_1")
        self.loss2 = Metrica(name="Loss_2")
        self.loss3 = Metrica(name="Loss_3")

    @property
    def metrics(self):
        return [self.total_loss, self.loss0, self.loss1,self.loss2,self.loss3]

    def compile_models(self):
        for model in [self.encoder, self.decoder, self.sindy]:
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3))
        self.compile(loss="mse",optimizer="sgd")

    def train_step(self,data):
        #data = tf.random.uniform((2,bs,Nsteps,128))
        x, x_dot = data #tf.random.uniform((2,bs,Nsteps,128))
        tf.print(x.shape)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.encoder.trainable_variables)
            tape.watch(self.decoder.trainable_variables)
            tape.watch(x)
            tape.watch(tf.convert_to_tensor(self.sindy.coeffs))

            ### LOSS 0###
            z = self.encoder(x)
            tape.watch(z)
            x_quasi = self.decoder(z)
            loss0 = tf.keras.losses.MSE(x,x_quasi)

            ### LOSS 1 ###
            zdot_SINDy = self.sindy(z)
            with tape.stop_recording():
                dpsi_dz = tape.jacobian(x_quasi,z)
            ones = tf.ones(dpsi_dz.shape)
            dpsi_dz = tf.einsum('abcdef,yuidep->abcf',dpsi_dz, ones) ### most of those derivatives are zero, since they correspond to different batches!
            xdot_pred = tf.einsum('ntaj,ntj->nta',dpsi_dz,zdot_SINDy)
            loss1 = tf.keras.losses.MSE(x_dot,xdot_pred)

            ### LOSS 2 ###
            with tape.stop_recording():
                dphi_dx = tape.jacobian(z,x)
            ones = tf.ones(dphi_dx.shape)
            dphi_dx = tf.einsum('abcdef,yuidep->abcf',dphi_dx, ones) ### most of those derivatives are zero, since they correspond to different batches!
            zdot_pred = tf.einsum('ntaj,ntj->nta',dphi_dx,x_dot)
            loss2 = tf.cast(tf.keras.losses.MSE(zdot_pred, zdot_SINDy),tf.float32)

            ### LOSS 3 ###
            loss3 = tf.expand_dims(tf.einsum('ij->',tf.math.abs(self.sindy.coeffs)),axis=0)
            total_loss = loss0 + loss1 + loss2 + loss3


        grads_enc = tape.gradient(total_loss, self.encoder.trainable_variables)
        grads_dec = tape.gradient(total_loss, self.decoder.trainable_variables)
        grads_SINDy_coeffs = [tape.gradient(total_loss, self.sindy.coeffs)]

        gradients = [grads_enc, grads_dec, grads_SINDy_coeffs]
        models = [self.encoder, self.decoder, self.sindy]

        for model, gradient in zip(models, gradients):
            model.optimizer.apply_gradients(zip(gradient, model.trainable_variables))

        self.total_loss.update_state(tf.reduce_mean(total_loss))
        self.loss0.update_state(tf.reduce_mean(loss0))
        self.loss1.update_state(tf.reduce_mean(loss1))
        self.loss2.update_state(tf.reduce_mean(loss2))
        self.loss3.update_state(tf.reduce_mean(loss3))

        return {k.name:k.result() for k in self.metrics}


class Metrica(tf.keras.metrics.Metric):
    def __init__(self, name):
        super(Metrica, self).__init__()
        self._name=name
        self.metric_variable = self.add_weight(name=name, initializer='zeros')

    def update_state(self, new_value, sample_weight=None):
        self.metric_variable.assign(new_value)

    def result(self):
        return self.metric_variable

    def reset_states(self):
        self.metric_variable.assign(0.)


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
