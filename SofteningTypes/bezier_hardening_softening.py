import tensorflow as tf

'''
    Functions for the Bezier hardening softening type
'''
def GetDamageThresholdBezierHardeningSoftening(equivalent_stress, damage_onset_stress):
    damage_threshold =  __damage_threshold(equivalent_stress, damage_onset_stress)
    return damage_threshold

def GetDamageVariableBezierHardeningSoftening(tau, e, e0, ei, ep, ej, ek, er, eu, \
                              s0, si, sp, sj, sk, sr, su):
    damage_threshold = __damage_threshold(tau, s0)
    damage_variable = __damage_variable_bezier_conditioning(\
                        damage_threshold, e, e0, ei, ep, ej, ek, er, eu, \
                                             s0, si, sp, sj, sk, sr, su)
    return damage_variable

'''
------------------------------------------------------------------------------
------------------------------------------------------------------------------
------------------------------------------------------------------------------
'''

def __damage_threshold(tau, f):
    with tf.name_scope('ComputationDamageThreshold'):
        r0 = tf.fill(tf.shape(tau), f)
        cond = tf.greater(tau, r0)
    r = tf.where(cond, tau, r0, name='DamageThreshold')
    return r

def __damage_variable_bezier_conditioning(r, e, e0, ei, ep, ej, ek, er, eu, \
                                             s0, si, sp, sj, sk, sr, su):
    with tf.name_scope("StrainLikePart"):
        Xi = __get_strain_like_counterpart(r,e)
    with tf.name_scope("BezConds"):
        c_lin, c_hard, c_soft1, c_soft2, c_resi = \
            __get_conditions_intervals(Xi, e0, ep, ek, eu)

        Xi_hard  = tf.where(c_hard, Xi, tf.zeros_like(Xi))
        Xi_soft1 = tf.where(c_soft1, Xi, tf.zeros_like(Xi))
        Xi_soft2 = tf.where(c_soft2, Xi, tf.zeros_like(Xi))

    with tf.name_scope("LinElaRange"):
        Bez_lin   = r
        d_lin   = __damage_variable_bezier(Bez_lin, r)
    with tf.name_scope("BezHardgRange"):
        Bez_hard  = __call_bezier_function(Xi_hard,e0,ei,ep,s0,si,sp)
        d_hard  = __damage_variable_bezier(Bez_hard, r)
    with tf.name_scope("BezSoftRange1"):
        Bez_soft1 = __call_bezier_function(Xi_soft1,ep,ej,ek,sp,sj,sk)
        d_soft1 = __damage_variable_bezier(Bez_soft1, r)
    with tf.name_scope("BezSoftRange2"):
        Bez_soft2 = __call_bezier_function(Xi_soft2,ek,er,eu,sk,sr,su)
        d_soft2 = __damage_variable_bezier(Bez_soft2, r)
    with tf.name_scope("ResiRange"):
        Bez_resi  = tf.fill(tf.shape(Xi),su)
        d_resi  = __damage_variable_bezier(Bez_resi, r)

    with tf.name_scope("BezierSelect"):
        d_tot = tf.where(c_lin, d_lin, \
                        tf.where(c_hard, d_hard, \
                                tf.where(c_soft1, d_soft1,\
                                        tf.where(c_soft2, d_soft2, d_resi)\
                    )))    
    return d_tot


def __get_strain_like_counterpart(r,e):
    XI = tf.divide(r,e,name='StrainLikeCounter')
    return XI

def __get_conditions_intervals(Xi, e0, ep, ek, eu):
    c_lin   = tf.less_equal(Xi,e0)
    c_hard  = tf.logical_and(tf.greater(Xi,e0), tf.less_equal(Xi,ep))
    c_soft1 = tf.logical_and(tf.greater(Xi,ep), tf.less_equal(Xi,ek))
    c_soft2 = tf.logical_and(tf.greater(Xi,ek), tf.less_equal(Xi,eu))
    c_resi  = tf.greater(Xi,eu)
    return (c_lin, c_hard, c_soft1, c_soft2, c_resi)

def __damage_variable_bezier(Bez, r):
    d = tf.subtract(1.0, tf.divide(Bez, r), name='DamageTemp')
    return d

def __call_bezier_function(X, x1, x2, x3, y1, y2, y3):
    with tf.name_scope('BezierConstants'):
        with tf.name_scope('A'):
            A = tf.subtract(tf.add(x1, x3), tf.multiply(2.0, x2), name="A")
            #x2 = tf.where(tf.less(A, 1.0e-12), x2 + 1.0e-6 * (x3-x1), x2)
        with tf.name_scope('B'):
            B = tf.multiply(2.0, tf.subtract(x2, x1), name="B")
        with tf.name_scope('C'):
            C = tf.subtract(x1, X, name = "C")
        with tf.name_scope('D'):
            D = tf.subtract(tf.square(B, name="B_squared"), tf.multiply(C, tf.multiply(A, 4.0, name = "4xA"), name = "4xAxC"), name="D_prev")
            Checker = tf.equal(X,tf.zeros_like(X))
            D = tf.where(Checker, tf.ones_like(D), D, name="D")
        with tf.name_scope('t'):
            t = tf.divide(tf.subtract(tf.sqrt(D), B), \
                            tf.multiply(2.0,A))
    with tf.name_scope('Bezier_Assembly'):
        Bez_1 = tf.multiply(tf.add(tf.subtract(y1, \
                                    tf.multiply(2.0, y2)), y3), \
                                    tf.pow(t,2.0))
        Bez_2 = tf.multiply(tf.multiply(2.0,t),tf.subtract(y2,y1))
    Bez = tf.add(tf.add(Bez_1,Bez_2),y1)
    return Bez