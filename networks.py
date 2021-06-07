import tensorflow as tf
from math import sqrt

"""
some notes:
(from Supplementar Information) which are applied at all15layers of the network, except for the last layer of the encoder and the last layer of the decoder


In all losses but the Loos0, you don't get gradients wrt decoder weigths... is this a bug? actually, in the paper they only compute wrt self.encoder: Can we give a reasonable explanation in terms of the chain rule? In terms of the "numerical part", the dependence is lost when you
apply the "tape.stop_recording(), which in turn allows us to compute the batched_jacobian"
"""

class MetaModel(tf.keras.Model):
    def __init__(self, models, lambda1=1e-4, lambda2=0,lambda3=1e-5, p_param=27, d_param=3, total_epochs=11000, when_zero_lambda3=1000,namerun="0"):
        """
        bs: batch_size
        Nt: time series length
        """
        super(MetaModel, self).__init__()
        
        self.namerun=namerun
        self.encoder, self.decoder, self.sindy = models
        #self.compile_models()
        
        self.total_epochs = total_epochs
        self.when_zero_lambda3 =  when_zero_lambda3
        self.total_loss = Metrica(name="Total Loss")
        self.loss0 = Metrica(name="Loss_0")
        self.loss1 = Metrica(name="Loss_1")
        self.loss2 = Metrica(name="Loss_2")
        self.loss3 = Metrica(name="Loss_3")
        self.p_param = p_param
        self.d_param = d_param

        ### regularizers
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    @property
    def metrics(self):
        """
        this helps monitring training
        """
        return [self.total_loss, self.loss0, self.loss1,self.loss2,self.loss3]

    def compile_models(self):
        """
        this internally defines, for each model, an optimizer.
        Importantly, you can access it through model.optimizer.
        We use this to "apply_gradients" in the train_step method
        """
        for model in [self.encoder, self.decoder, self.sindy]:
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3))
        self.compile(loss="mse",optimizer="sgd") #this is required to use model.fit, in the MetaModel; but "sgd" and "mse" does not matter

    def train_step(self,data):
        x, x_dot = data #DATA of shape (BS,128)

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.encoder.trainable_variables)
            tape.watch(self.decoder.trainable_variables)
            tape.watch(x)
            tape.watch(self.sindy.trainable_variables)

            ### LOSS 0###
            z = self.encoder(x)
            tape.watch(z)
            x_quasi = self.decoder(z)
            loss0 = tf.keras.losses.MSE(x,x_quasi)

            ### LOSS 1 ###
            with tape.stop_recording():
                dpsi_dz = tape.batch_jacobian(x_quasi,z) # (this has dimenison BATCH, 128, 3)


            zdot_SINDy = self.sindy(z)
            xdot_pred = tf.einsum('bxz,bz->bx',dpsi_dz,zdot_SINDy)

            loss1 = self.lambda1*tf.keras.losses.MSE(x_dot,xdot_pred)

            ### LOSS 2 ###
            with tape.stop_recording():
                dphi_dx = tape.batch_jacobian(z,x) #this has dimension BATCH, 3, 128
            zdot_pred = tf.einsum('bjx,bx->bj',dphi_dx,x_dot)
            loss2 = self.lambda2*tf.keras.losses.MSE(zdot_pred, zdot_SINDy)

            ### LOSS 3 ###
            loss3 = self.lambda3*tf.expand_dims(tf.einsum('ij->',tf.math.abs(self.sindy.coeffs)),axis=0)/(self.p_param*self.d_param)
            total_loss = loss0 + loss1 + loss2 + loss3


        grads_enc = tape.gradient(total_loss, self.encoder.trainable_variables)
        grads_dec = tape.gradient(total_loss, self.decoder.trainable_variables)
        grads_SINDy_coeffs = tape.gradient(total_loss, self.sindy.trainable_variables)

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
    """
    This helps to monitor training (for instance each loss)
    """
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

class TrainingCallback(tf.keras.callbacks.Callback):
    '''Stop training when enough time has passed.

        # Arguments
        seconds: maximum time before stopping.
        verbose: verbosity mode.
    '''
    def __init__(self):
        super(TrainingCallback, self).__init__()

    def on_epoch_begin(self, epoch, logs={}):
        if (epoch >  self.model.total_epochs - self.model.when_zero_lambda3) and (self.model.total_epochs > 9*1e3):
            self.model.lambda3 = 0
        elif epoch%500 == 0:
            x = self.model.sindy.coeffs
            if epoch>500:
                self.model.sindy.coeffs = tf.where( tf.abs(x) > 0.1, x, 0)
            for name, model in zip(["encoder","decoder","sindy"],[self.model.encoder, self.model.decoder, self.model.sindy]):
                model.save_weights("/data/uab-giq/scratch/matias/sandra/networks/run{}/{}_{}/".format(self.model.namerun,name,epoch))


class Encoder(tf.keras.Model):
    def __init__(self, seed_val=0.1):
        """
        Encoder network
        """
        super(Encoder,self).__init__()
        alphaxavi = sqrt(6)/(128+3)
        self.l1 = tf.keras.layers.Dense(64,kernel_initializer=tf.random_uniform_initializer(minval=-alphaxavi, maxval=alphaxavi),
                    bias_initializer = tf.keras.initializers.Zeros())
        self.l2 = tf.keras.layers.Dense(32,kernel_initializer=tf.random_uniform_initializer(minval=-alphaxavi, maxval=alphaxavi),
                    bias_initializer = tf.keras.initializers.Zeros())
        self.loutput = tf.keras.layers.Dense(3,kernel_initializer=tf.random_uniform_initializer(minval=-alphaxavi, maxval=alphaxavi),
                    bias_initializer = tf.keras.initializers.Zeros())

    def call(self, inputs):
        f = tf.nn.sigmoid(self.l1(inputs))
        f = tf.nn.sigmoid(self.l2(f))
        f = self.loutput(f)
        return f


class Decoder(tf.keras.Model):
    def __init__(self, seed_val=0.1):
        """
        Decoder network
        """
        super(Decoder,self).__init__()
        alphaxavi = sqrt(6)/(3+128)
        self.l1 = tf.keras.layers.Dense(32,kernel_initializer=tf.random_uniform_initializer(minval=-alphaxavi, maxval=alphaxavi),
                    bias_initializer = tf.keras.initializers.Zeros())
        self.l2 = tf.keras.layers.Dense(64,kernel_initializer=tf.random_uniform_initializer(minval=-alphaxavi, maxval=alphaxavi),
                    bias_initializer = tf.keras.initializers.Zeros())
        self.loutput = tf.keras.layers.Dense(128,kernel_initializer=tf.random_uniform_initializer(minval=-alphaxavi, maxval=alphaxavi),
                    bias_initializer = tf.keras.initializers.Zeros())

    def call(self, inputs):
        f = tf.nn.sigmoid(self.l1(inputs))
        f = tf.nn.sigmoid(self.l2(f))
        f = self.loutput(f)
        return f
