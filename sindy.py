import numpy as np
import ast
import tensorflow as tf


class SINDy_tf():
    def __init__(self):
        self.coeffs = tf.ones((3,27))

    def theta(self,z):
        th = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    th.append(tf.pow(z[0], i)*tf.pow(z[1], j)*tf.pow(z[2], k))
        return tf.transpose(tf.stack(th,axis=0))

    def __call__(self, phi_of_ex):
        thetas = self.theta(phi_of_ex)
        zdot_sindy = tf.einsum('ij,kj->ki',self.coeffs,thetas)
        return zdot_sindy





class SINDy():
    def __init__(self):
        self.sindy_coeffs, self.list_to_index, self.index_to_list  = self.initialize_dictionaries()

    def initialize_dictionaries(self):
        """
        coeffs: first 3 comes from number of z components, the remaining is determined by p,
        the number of theta functions, each theta encoded in a list that determines the polynomial.
        """
        coeffs = np.ones((3,)+tuple([3]*3))
        ind=0
        list_to_index={}
        index_to_list={}
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    lista=str([i,j,k])
                    list_to_index[lista]=ind
                    index_to_list[ind]=lista
                    ind+=1
        return coeffs, list_to_index, index_to_list

    def theta_fun(self,indices, zetas):
        """
        deliver z1^n1 z2^n2 z3^n3
        SINDy library polynomial order 3.
        """
        if isinstance(indices,str):
            indices = ast.literal_eval(indices)
        val=1
        for exp, z in zip(indices, zetas):
            val*=np.power(z,exp)
        return val

    def __call__(self, encoder_output, zdot_comp):
        """
        encoder_output: array of dimension 3 (for Lorenz case)
        zdot_comp: int <3. Is the component of the zdot (which determines the row of the SINDy coefficient matrix)
        """
        sindy_coeffs = []
        theta_values = []
        for ind in range(27):
            index_sindys = tuple([zdot_comp]) + tuple(ast.literal_eval(self.index_to_list[ind]))
            sindy_coeffs.append(self.sindy_coeffs[index_sindys])
            theta_values.append(self.theta_fun(self.index_to_list[ind],encoder_output))
        return sindy_coeffs, theta_values
        # return np.dot(sindy_coeffs, theta_values)
