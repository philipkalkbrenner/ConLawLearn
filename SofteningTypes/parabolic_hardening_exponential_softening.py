import tensorflow as tf

'''
    Functions for the parabolic hardening & exponential softening type
'''
def GetDamageThresholdParabolicHardeningExponentialSoftening(equivalent_stress, damage_onset_stress):
    damage_threshold =  __damage_threshold(equivalent_stress, damage_onset_stress)
    return damage_threshold

def GetDamageVariableParabolicHardeningExponentialSoftening(\
        equivalent_stress, young_modulus, yield_stress, fracture_energy, \
        damage_onset_threshold, yield_threshold, bounding_damage_threshold):
    damage_threshold = __damage_threshold(equivalent_stress, damage_onset_threshold)
    
    damage_variable_hardening = __damage_variable_par_hard(damage_threshold, damage_onset_threshold, yield_threshold, bounding_damage_threshold)
    softening_parameter = __discrete_softening_parameter_par_hard_exp_soft(\
                            damage_onset_threshold, yield_threshold, bounding_damage_threshold,\
                            young_modulus, yield_stress, fracture_energy)
    damage_variable_softening =  __damage_variable_par_hard_expo_soft(damage_threshold, yield_threshold, \
                                    bounding_damage_threshold, softening_parameter)
    damage_variable = __damage_variable_selection_par_hard_exp_soft(\
                        damage_threshold, damage_variable_hardening, \
                        damage_variable_softening, damage_onset_threshold, bounding_damage_threshold)
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

def __damage_variable_par_hard(r, r0, re, rp):
    with tf.name_scope('D_ParHard'):
        ad = __get_Ad_parameter(rp, re)
        d_temp1 = tf.square(tf.divide(tf.subtract(r, r0), \
                                tf.subtract(rp, r0)))
        
        d_temp2 = tf.multiply(tf.divide(re, r), ad)
        d = tf.multiply(d_temp1, d_temp2, name='D_ParHard')
    return d

def __damage_variable_par_hard_expo_soft(r, re, rp, hd):
    with tf.name_scope('D_ParHardExpSoft'):
        d_temp1 = tf.multiply(\
            tf.multiply(2.0, hd),tf.divide(tf.subtract(rp, r), re))
        d_temp2 = tf.exp(d_temp1)
        d_temp3 = tf.multiply(d_temp2, tf.divide(re, r))
        d = tf.subtract(1.0, d_temp3, name='D_PHExpSoft')
    return d

def __discrete_softening_parameter_par_hard_exp_soft(r0, re, rp, e, f, g):
    with tf.name_scope('Hd_Par_Hard_Exp_Soft'):
        ad_snake = __get_Ad_snake_parameter(rp, r0, re)
        temp1 = tf.multiply(tf.divide(tf.multiply(2.0, e), \
                                            tf.square(f)), g)
        temp2 = tf.divide(rp, re)
        temp3 = tf.subtract(tf.subtract(temp1, ad_snake),temp2)
        h = tf.divide(1.0,temp3, name='Hd')
    return h

def __damage_variable_selection_par_hard_exp_soft(r, d_hard, d_soft, r0, rp):
    with tf.name_scope('D_ParHardExpSoft_Selection'):
        R0 = tf.fill(tf.shape(r), r0)
        Rp = tf.fill(tf.shape(r), rp)
        cond1 = tf.greater(r, Rp)
        cond2 = tf.greater(r, R0)
    d = tf.where(cond1, d_soft, tf.where(cond2, d_hard, tf.zeros_like(r))\
        ,name='D_ParHardExpSoft')
    return d

def __get_Ad_snake_parameter(rp,r0,re):
    with tf.name_scope('Ad_snake'):
        ad = __get_Ad_parameter(rp,re)
        temp1 = tf.multiply(2.0, tf.pow(r0,3))
        temp2 = tf.multiply(3.0, tf.multiply(rp, tf.square(r0)))
        temp3 = tf.pow(rp, 3)
        temp4 = tf.add(tf.subtract(temp3, temp2), temp1)
        temp5 = tf.multiply(ad, temp4)
        temp6 = tf.square(tf.subtract(rp, r0))
        temp7 = tf.multiply(3.0, tf.multiply(re, temp6))
        ad_snake = tf.divide(temp5, temp7, name="Ad_snake")
    return ad_snake

def __get_Ad_parameter(rp,re):
    with tf.name_scope('Ad'):
        ad = tf.divide(tf.subtract(rp, re), re)
    return ad




