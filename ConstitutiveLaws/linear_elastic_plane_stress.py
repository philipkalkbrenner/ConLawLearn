import tensorflow as tf

'''
The class to call the constitutive law for the:
    LINEAR ELASTIC PLANE STRESS
'''

class LinearElasticPlaneStress(object):
    def __init__(self, variables):
        self.e = variables['E']
        self.nu = variables['NU']

    def GetEffectiveStress(self, strain_vector):
        with tf.name_scope("LinearElasticLawPlaneStress"):
            with tf.name_scope("ElasticityTensor"):
                elasticity_matrix = self.__elastictiy_tensor_plane_stress()
            with tf.name_scope("EffectiveStressVector"):
                stress_vector = tf.matmul(eps, elasticity_matrix)
        return stress_vector
    
    def GetLinearElasticStress(self, strain_vector):
        with tf.name_scope("LinearElasticLawPlaneStress"):
            with tf.name_scope("ElasticityTensor"):
                elasticity_matrix = self.__elastictiy_tensor_plane_stress()
            with tf.name_scope("LinearElasticStressVector"):
                stress_vector = tf.matmul(strain_vector, elasticity_matrix)
        return stress_vector
    
    def GetStress(self, strain_vector):
        with tf.name_scope("LinearElasticLawPlaneStress"):
            with tf.name_scope("ElasticityTensor"):
                elasticity_matrix = self.__elastictiy_tensor_plane_stress()
            with tf.name_scope("LinearElasticStressVector"):
                stress_vector = tf.matmul(strain_vector, elasticity_matrix)
        return stress_vector

    '''
    Internal Functions
    '''

    def __elastictiy_tensor_plane_stress(self):
        pos_11 = tf.divide(self.e,tf.subtract(1.0,tf.square(self.nu)))
        pos_12 = tf.divide(tf.multiply(self.e, self.nu),tf.subtract(1.0,tf.square(self.nu)))
        pos_33 = tf.divide(self.e,tf.multiply(2.0,tf.add(1.0,self.nu)))
        
        c_temp = [    [pos_11   ,   pos_12  ,   0.0     ]     , \
                      [pos_12   ,   pos_11  ,   0.0     ]     , \
                      [0.0      ,   0.0     ,   pos_33  ]   ]
        c = tf.stack(c_temp)
        return c
        
