import tensorflow as tf

'''
    Functions for the exponential softening type
'''
def GetDamageThresholdExponentialSoftening(equivalent_stress, yield_stress):
    damage_threshold, initial_damage_threshold = __damage_threshold(equivalent_stress, yield_stress)
    return damage_threshold

def GetDamageVariableExponentialSoftening(equivalent_stress, young_modulus, yield_stress, fracture_energy):
    damage_threshold, initial_damage_threshold = __damage_threshold(equivalent_stress, yield_stress)
    softening_parameter = __discrete_softening_parameter_exp_soft(young_modulus, yield_stress, fracture_energy)
    damage_variable = __damage_variable_exp_soft(damage_threshold, initial_damage_threshold, softening_parameter)     
    return damage_variable

'''
------------------------------------------------------------------------------
'''
def __damage_threshold(tau, f):
    with tf.name_scope('ComputationDamageThreshold'):
        r0 = tf.fill(tf.shape(tau), f)
        cond = tf.greater(tau, r0)
    r = tf.where(cond, tau, r0, name='DamageThreshold')
    return (r, r0)

def __damage_variable_exp_soft(r, r0, hd):
    with tf.name_scope('Comp_D_ExpoSoft'):
        d_temp1 = tf.divide(r0,r)
        d_temp2 = tf.multiply(2.0, tf.multiply(hd, \
                                tf.divide(tf.subtract(r0,r), r0)))
        d_temp3 = tf.exp(d_temp2)
        d_temp4 = tf.multiply(d_temp1, d_temp3)
    d = tf.subtract(1.0,d_temp4, name="Damagevariable")
    return d

def __discrete_softening_parameter_exp_soft(e, f, g):
    with tf.name_scope('Hd_Exp_Soft'):
        h_temp1 = tf.multiply(tf.divide(tf.multiply(2.0, e), \
                                        tf.square(f)),g)
        h_temp2 = tf.subtract(h_temp1, 1.0)
        h =tf.divide(1.0, h_temp2, name='Hd')
    return h