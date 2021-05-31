with tf.GradientTape(persistent=True) as tape:
    tape.watch(encoder.trainable_variables)
    tape.watch(decoder.trainable_variables)
    tape.watch(x)
    tape.watch(sindy.trainable_variables)

    ### LOSS 0###
    z = encoder(x)
    tape.watch(z)
    x_quasi = decoder(z)
    loss0 = tf.keras.losses.MSE(x,x_quasi)

    ### LOSS 1 ###
    with tape.stop_recording():
        dpsi_dz = tape.batch_jacobian(x_quasi,z) # (this has dimenison BATCH, 128, 3)
        ##### CAREFUL BECAUSE HERE IT'S NOT RECORDING FOR THE DECODER... is this a problem ? in the paper they onyl compute wrt encoder, it seems.

    zdot_SINDy = sindy(z)
    xdot_pred = tf.einsum('bxz,bz->bx',dpsi_dz,zdot_SINDy)

    loss1 = lambda1*tf.keras.losses.MSE(x_dot,xdot_pred)

    ### LOSS 2 ###
    with tape.stop_recording():
        dphi_dx = tape.batch_jacobian(z,x)
    zdot_pred = tf.einsum('bjx,bx->bj',dphi_dx,x_dot)
    loss2 = lambda2*tf.keras.losses.MSE(zdot_pred, zdot_SINDy)

    ### LOSS 3 ###
    loss3 = lambda3*tf.expand_dims(tf.einsum('ij->',tf.math.abs(sindy.coeffs)),axis=0)/(p_param*d_param)
    total_loss = loss0 + loss1 + loss2 + loss3
