import tensorflow as tf

'''
    Functions for the linear softening type
'''
def GetDamageThresholdLinearSoftening(equivalent_stress, yield_stress):
    damage_threshold = __damage_threshold(equivalent_stress, yield_stress)
    return damage_threshold

def GetDamageVariableLinearSoftening(equivalent_stress, yield_stress, bounding_damage_threshold):
    damage_threshold = __damage_threshold(equivalent_stress, yield_stress)
    damage_variable = __damage_variable_lin_soft(damage_threshold, yield_stress, bounding_damage_threshold)     
    return damage_variable

'''
------------------------------------------------------------------------------
'''
def __damage_threshold(tau, f):
    with tf.name_scope('ComputationDamageThreshold'):
        r0 = tf.fill(tf.shape(tau), f)
        cond = tf.greater(tau, r0)
    r = tf.where(cond, tau, r0, name='DamageThreshold')
    return r

def __damage_variable_lin_soft(r, rp, ru):
    with tf.name_scope('Comp_D_LinSoft'):
        d_temp1 = tf.multiply(rp, tf.subtract(r,ru))
        d_temp2 = tf.subtract(ru,rp)
        d_temp3 = tf.subtract(rp,tf.divide(d_temp1, d_temp2))
        d_temp4 = tf.multiply(tf.divide(1.0,r), d_temp3) 

    d = tf.subtract(1.0,d_temp4, name="Damagevariable")
    return d