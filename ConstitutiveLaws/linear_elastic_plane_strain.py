import tensorflow as tf

'''
The class to call the constitutive law for the:
    LINEAR ELASTIC PLANE STRAIN
'''

class LinearElasticPlaneStrain(object):
    def __init__(self, variables):
        self.e = variables['E']
        self.nu = variables['NU']
        
    def GetEffectiveStress(self, strain_vector):
        stress_vector = self.GetStress(strain_vector)
        return stress_vector

    def GetLinearElasticStress(self, strain_vector):
        stress_vector = self.GetStress(strain_vector)
        return stress_vector

    def GetStress(self, strain_vector):
        with tf.name_scope("LinearElasticLawPlaneStrain"):
            with tf.name_scope("ElasticityTensor"):
                elasticity_matrix = self.__elastictiy_tensor_plane_strain()
            with tf.name_scope("EffectiveStressVector"):
                stress_vector = tf.matmul(strain_vector, elasticity_matrix)
        return stress_vector

    '''
    Internal Functions
    '''

    def __elastictiy_tensor_plane_strain(self):
        pos_11 = tf.divide( \
            tf.multiply(tf.subtract(1.0,self.nu),self.e), \
            tf.multiply(tf.add(1.0,self.nu), \
                        tf.subtract(1.0,tf.multiply(2.0,self.nu) ) ))
        pos_12 = tf.divide( \
            tf.multiply(self.nu,self.e), \
            tf.multiply(tf.add(1.0,self.nu), \
                        tf.subtract(1.0,tf.multiply(2.0,self.nu) ) ))
        pos_33 = tf.divide(self.e,tf.multiply(2.0,tf.add(1.0,self.nu)))
        c_temp = [[pos_11, pos_12, 0.0     ], \
                  [pos_12, pos_11, 0.0     ],\
                  [0.0     , 0.0     , pos_33]]
        c = tf.stack(c_temp)
        return c
        
