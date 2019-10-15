import numpy as np
import tensorflow as tf

class EffectiveStressSplit(object):
    
    def GetPositiveStress(sig_eff):
        (sig_eff_pos, garbage) = EffectiveStressSplit.__split_effective_stress(sig_eff)
        sig_eff_pos = tf.where(tf.less(tf.abs(sig_eff_pos),1e-12), tf.zeros_like(sig_eff_pos), sig_eff_pos)
        return sig_eff_pos

    def GetNegativeStress(sig_eff):
        (garbage, sig_eff_neg) = EffectiveStressSplit.__split_effective_stress(sig_eff)
        sig_eff_neg = tf.where(tf.less(tf.abs(sig_eff_neg),1e-12), tf.zeros_like(sig_eff_neg), sig_eff_neg) 
        return sig_eff_neg

    def GetPrincipalDirection(sig_eff):
        Theta = EffectiveStressSplit.__get_principal_direction(sig_eff)
        return Theta

    #def Get




    def __split_effective_stress(sig_eff):
        batch = tf.cast(tf.shape(sig_eff)[0],tf.int32)
        SIG_EFF = EffectiveStressSplit.__voigt_to_matrix_2d(batch, sig_eff)
        THETA = EffectiveStressSplit.__get_principal_direction(sig_eff)
        R,RT  = EffectiveStressSplit.__get_rotation_matrix(THETA)
        SIG_EFF_PRI = tf.matmul(tf.matmul(R, SIG_EFF), RT)
        S1 = SIG_EFF_PRI[:,0][:,0]
        S2 = SIG_EFF_PRI[:,1][:,1]

        with tf.name_scope('PositivePart'):
            cond = tf.greater(SIG_EFF_PRI,0.0)
            POS_PRI = tf.where(cond, SIG_EFF_PRI, tf.zeros([batch,2,2]))
            SIG_EFF_POS = tf.matmul(tf.matmul(RT, POS_PRI), R)
        
            with tf.name_scope('VoigtPosPrincipalStates'):
                pos_part_temp = tf.fill([batch,3],0.0)
                pos_1 = tf.add(pos_part_temp[:,0],SIG_EFF_POS[:,0,0])
                pos_2 = tf.add(pos_part_temp[:,0],SIG_EFF_POS[:,1,1])
                pos_3 = tf.add(pos_part_temp[:,0],SIG_EFF_POS[:,0,1])
            with tf. name_scope('SigmaEffPos'):
                sig_eff_pos_temp  = tf.stack([pos_1, pos_2, pos_3], axis=1)
        with tf.name_scope('NegativePart'):
            with tf.name_scope('SigmaEffNeg'):
                sig_eff_neg_temp = tf.subtract(sig_eff, sig_eff_pos_temp)

            sig_eff_zeros = tf.zeros_like(sig_eff)    
            zeros = tf.zeros_like(S1)
            cond_tot_pos = tf.logical_and(tf.greater_equal(S1, zeros), tf.greater_equal(S2,zeros))
            cond_tot_neg = tf.logical_and(tf.less_equal(S1, zeros), tf.less_equal(S2,zeros))
            sig_eff_neg = tf.where(cond_tot_pos, sig_eff_zeros, sig_eff_neg_temp)
            sig_eff_pos = tf.where(cond_tot_neg, sig_eff_zeros, sig_eff_pos_temp)
            
        return (sig_eff_pos, sig_eff_neg)

    def __voigt_to_matrix_2d(batch, matrix):
        with tf.name_scope('InputReverse'):
            dims = tf.constant([1])
            reverse_matrix = tf.reverse(matrix,dims)
        with tf.name_scope('FirstRow'):
            first_row_temp1 = matrix[:,:1]
            first_row_temp2 = reverse_matrix[:,:1]
            first_row = tf.concat([first_row_temp1, first_row_temp2],1, \
                              name='Row1')
        with tf.name_scope('SecondRow'):
            second_row = reverse_matrix[:,:-1]
        with tf.name_scope('Matrix2X2'):
            final_matrix = tf.reshape(tf.concat([first_row,second_row],1), \
                                  [batch,2,2])
        return final_matrix

    def __get_principal_direction_old(eps):
        with tf.name_scope('Computation_D'):
            with tf.name_scope('Size'):
                batch = tf.cast(tf.shape(eps)[0],tf.int32) # shape of matrix
            with tf.name_scope('Numerator'):
                numerator = eps[:,2]
            with tf.name_scope('Denomimator'):
                denominator = tf.subtract(eps[:,0],eps[:,1])
            with tf.name_scope('Selection'):
                
                argument = EffectiveStressSplit.__case_selection(batch, numerator, denominator)
        with tf.name_scope('Theta_Principal'):
            angle = tf.multiply(tf.constant(0.50),argument, name='Theta_P')
        return angle
    
    def __get_principal_direction(sig_eff):
        with tf.name_scope('Computation_E'):
            with tf.name_scope('Size'):
                batch = tf.cast(tf.shape(sig_eff)[0],tf.int32) # shape of matrix
            with tf.name_scope('Numerator'):
                numerator = tf.multiply(2.0, sig_eff[:,2])
            with tf.name_scope('Denomimator'):
                denominator = tf.subtract(sig_eff[:,0],sig_eff[:,1])
            with tf.name_scope('Selection'):
                argument = EffectiveStressSplit.__case_selection(batch, numerator, denominator)
        with tf.name_scope('Theta_Principal'):
            angle = tf.multiply(tf.constant(0.50),argument, name='Theta_P')
        return angle
        '''

    def __case_selection(batch, num, den):
        cond_num = tf.less(tf.abs(num),1e-12, name='CondNumerator')
        cond_den = tf.less(tf.abs(den),1e-12, name='CondDenominator')
        cond_num_den = tf.logical_and(tf.less(tf.abs(num),1e-12), tf.less(tf.abs(den),1e-12))
        pi_half  = tf.fill([batch],tf.divide(np.pi,2.0), name='PiHalf')
        den = tf.where(cond_num_den, tf.ones_like(den), den)
        inner_where = tf.where(cond_den, pi_half, \
                        tf.atan(tf.divide(num,den)), name='Inner_Request')
        arg = tf.where(cond_num,tf.zeros([batch]),inner_where, \
                            name='TanArgu')
        return arg
        '''
    
    def __case_selection(batch, num, den):
        num_abs = tf.abs(num)
        den_abs = tf.abs(den)
        tolerance = 1.0e-12

        cond_num_less = tf.less_equal(num_abs, tolerance)
        cond_num_greater = tf.greater(num_abs, tolerance)
        cond_den_less = tf.less_equal(den_abs, tolerance)
        cond_den_greater = tf.greater(den_abs, tolerance) 

        # num = 0 and den = 0:
        cond_num_den_1 = tf.logical_and(cond_num_less, cond_den_less)
        # num = 0 and den =! 0:
        cond_num_den_2 = tf.logical_and(cond_num_less, cond_den_greater)
        # num =! 0 and den = 0
        cond_num_den_3 = tf.logical_and(cond_num_greater, cond_den_less)
        # num =! 0 and den =! 0:
        cond_num_den_4 = tf.logical_and(cond_num_greater, cond_den_greater)

        num = tf.where(cond_num_den_4, num, tf.zeros_like(num))
        den = tf.where(cond_num_den_4, den, tf.ones_like(den))

        pi_half  = tf.fill([batch],tf.divide(np.pi,2.0), name='PiHalf')

        arg = tf.where(cond_num_den_4, x = tf.atan(tf.divide(num,den)), y = tf.where(cond_num_den_3, x = pi_half, y = tf.zeros_like(num)))
        return arg

    def __get_rotation_matrix(angle):
        with tf.name_scope('RotationMatrix'):
            with tf.name_scope('Computation_F'):
                sin = tf.expand_dims(tf.sin(angle),1, name='Sine')
                cos = tf.expand_dims(tf.cos(angle),1,name='Cosine')
                n1 = tf.concat([cos,sin],1, name='Vector1')
                n2 = tf.concat([tf.multiply(sin,-1.0),cos],1, name='Vector2')
                R_temp1= tf.expand_dims(n1,1)
                R_temp2= tf.expand_dims(n2,1)
            with tf.name_scope('R'):
                R = tf.concat([R_temp1,R_temp2],1, name='R')
            with tf.name_scope('R_T'):
                RT = tf.transpose(R,perm=[0,2,1], name='R_T')
        return R, RT

    
