def train_step(self,data):
    #data = tf.random.uniform((2,bs,Nsteps,128))

    x, x_dot = data #tf.random.uniform((2,bs,Nsteps,128))
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
    grads_SINDy_coeffs = tape.gradient(total_loss, self.sindy.coeffs)

    gradients = [grads_enc, grads_dec, grads_SINDy_coeffs]
    models = [self.encoder, self.decoder, self.sindy]

    for model, gradient in zip(models, gradients):
        model.optimizer.apply_gradients(zip(gradient, model.trainable_variables))
    losses = tf.stack([loss0,loss1,loss2,loss3,loss4])
    return total_loss, losses
