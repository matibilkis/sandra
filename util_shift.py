

self.mask = tf.keras.layers.Masking(mask_value=pad_value,
                                    input_shape=(None, 2))
                                  # input_shape=(self.dolinar_layers, 2)) #(beta1, pad), (n1, beta2), (n2, guess). In general i will have (layer+1)



    def train_step(self, data):
        x,y=data
        with tf.GradientTape() as tape:
            tape.watch(self.trainable_variables)
            preds = self(x,training=True)
            energy = self.compiled_loss(preds, preds) #notice that compiled loss takes care only about the preds
        train_vars = self.trainable_variables
        grads=tape.gradient(energy,train_vars)
        self.gradient_norm.update_state(tf.reduce_sum(tf.pow(grads[0],2)))

        if self.optimizer.get_config()["name"] == "SGD":
            self.qacq_gradients(energy,grads,x)
        else:
            self.optimizer.apply_gradients(zip(grads, train_vars))
        self.cost_value.update_state(energy)
        self.lr_value.update_state(self.optimizer.lr)
        return {k.name:k.result() for k in self.metrics}





















    for i in range(3):
        for j in range(3):
            for k in range(3):
                lista=str([i,j,k])
                list_to_index[lista]=ind
                index_to_list[ind]=lista
                ind+=1















### LOSS 0 and 1 ###
with tf.GradientTape(persistent=True) as tape0:
    tape0.watch(x)
    tape0.watch(encoder.trainable_variables)
    tape0.watch(decoder.trainable_variables)
    z = encoder(x)
    tape0.watch(z)
    x_quasi = decoder(z)

loss0 = tf.keras.losses.MSE(x,x_quasi)

zdot_sindy = tf.cast(tf.expand_dims(np.array([sindy(np.squeeze(z),k) for k in range(3)]), axis=0),tf.float32)
dpsi_dz = tf.squeeze(tape0.jacobian(x_quasi, z))
xdot_pred = tf.matmul(zdot_sindy,dpsi_dz, transpose_b=True)
loss1 = tf.keras.losses.MSE(x_dot,xdot_pred )

### LOSS 2 and 4 ###

dphi_dx = tf.squeeze(tape0.jacobian(z,x))
#tf.einsum('ij,j->i', dphi_dx, tf.squeeze(x_dot))
zdot_pred = tf.matmul(x_dot, dphi_dx, transpose_b=True)
encoder_output = np.squeeze(encoder(x))
zdot_sindy = tf.expand_dims(np.array([sindy(encoder_output,k) for k in range(3)]), axis=0)
loss2 = tf.cast(tf.keras.losses.MSE(zdot_pred, zdot_sindy),tf.float32)

with tf.GradientTape(persistent=True) as tape2:
    tape2.watch(x)
    z = encoder(x)
dphi_dx = tf.squeeze(tape2.jacobian(z,x))
#tf.einsum('ij,j->i', dphi_dx, tf.squeeze(x_dot))
zdot_pred = tf.matmul(x_dot, dphi_dx, transpose_b=True)
loss3 = tf.cast(tf.keras.losses.MSE(zdot_pred, zdot_sindy),tf.float32)

loss4 = tf.cast(tf.expand_dims(np.sum(tf.abs(sindy.sindy_coeffs)),axis=0),tf.float32)





###
tf.matmul(thetas,coeffs , transpose_b=True)
