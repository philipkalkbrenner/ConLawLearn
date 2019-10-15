import tensorflow as tf
import numpy as np

class YieldCriteriaUtilities(object):

    def Kb(fcbi, fcp):
        kb = tf.divide(fcbi,fcp, name="kb")
        return kb
    
    def Alpha(kb):
        alpha = tf.divide(tf.subtract(kb, 1.0), \
                          tf.subtract(tf.multiply(2.0, kb), 1), name= "alpha")
        return alpha
    
    def Beta(fcp, ft, alpha):
        b_temp1 = tf.multiply(tf.divide(fcp, ft),tf.subtract(1.0,alpha))
        b_temp2 = tf.add(1.0,alpha)
        beta = tf.subtract(b_temp1, b_temp2, name="beta")
        return beta

    def FirstInvariant(sig_eff):
        with tf.name_scope('1stInvariant'):
            i1 = tf.add(sig_eff[:,0], sig_eff[:,1], name="I1")
        return i1

    def SecondDeviatoricInvariant(sig_eff):
        with tf.name_scope('2ndDevInvariant'):
            batch = tf.cast(tf.shape(sig_eff)[0],tf.int32)
            with tf.name_scope('Computation_H'):
                j2_sum1 = tf.pow(sig_eff[:,0],2, name = "j2_sum1")
                j2_sum2 = tf.multiply(sig_eff[:,0], sig_eff[:,1], name = "j2_sum2")
                j2_sum3 = tf.pow(sig_eff[:,1],2, name = "j2_sum3")
                j2_sum4 = tf.multiply(3.0,tf.pow(sig_eff[:,2],2), name="j2_sum4")

                j2 = tf.divide(tf.subtract(tf.add(tf.add(j2_sum1, j2_sum3),\
                                        j2_sum4),j2_sum2),3.0, name="J2")
        return j2

    def PrincipalStressValues(sig_eff):
        sig_eff_pri = YieldCriteriaUtilities.__principal_stress_matrix(sig_eff)
        with tf.name_scope('PrincStresses'):
            with tf.name_scope('Sigma1'):
                s1 = sig_eff_pri[:,0][:,0]
            with tf.name_scope('Sigma2'):
                s2 = sig_eff_pri[:,1][:,1]
        return s1, s2
    
    def PrincipalStressMax(s1, s2):
        '''
            Returns the maximum value of s1 or s2 and only if greater than zero, else returns zero
        '''
        with tf.name_scope('S_max'):
            zeros = tf.zeros(tf.shape(s1))
            s_max = tf.where(tf.greater(s1, s2), s1, s2)
            s_max_macaulay = tf.where(tf.greater_equal(s_max, zeros), s_max, zeros, name = "s_max_macaulay")
        return s_max_macaulay

    def PrincipalStressMaxHeaviside(s1,s2):
        zeros = tf.zeros(tf.shape(s1))
        ones  = tf.add(zeros, 1.0)
        s_max = tf.where(tf.greater(s1, s2), s1, s2)
        smax_heavi = tf.where(tf.greater(s_max,zeros), ones, zeros)
        return smax_heavi

    def PrincipalStressMin(s1,s2):
        '''
            Returns the minimum value of s1 or s2 and only if less than zero, else returns zero
        '''
        with tf.name_scope('S_min'):
            zeros = tf.zeros(tf.shape(s1))
            s_min = tf.where(tf.less(s1, s2), s1, s2)
            s_min_macaulay = tf.where(tf.greater_equal(s_min, zeros), zeros, s_min, name = "s_max_macaulay")
        return s_min

    def __get_equivalent_stress_smin(s1, s2):
        with tf.name_scope('S_min'):
            zeros = tf.zeros(tf.shape(s1))
            s_min = tf.where(tf.less(s1, s2), s1, s2)
        return s_min

    def PrincipalStressMinHeaviside(s1,s2):
        zeros = tf.zeros(tf.shape(s1))
        ones  = tf.add(zeros, 1.0)
        s_min = tf.where(tf.less(s1, s2), s1, s2)
        s_min_heavi = tf.where(tf.greater(s_min,zeros), zeros, ones)
        return s_min_heavi

    def __principal_stress_matrix(sig_eff):
        batch = tf.cast(tf.shape(sig_eff)[0],tf.int32)
        SIG_EFF = YieldCriteriaUtilities.__get_voigt_to_matrix_2d(batch, sig_eff)
    
        THETA = YieldCriteriaUtilities.__get_principal_direction(sig_eff)
        R,RT  = YieldCriteriaUtilities.__get_rotation_matrix(THETA)
        SIG_EFF_PRI = tf.matmul(tf.matmul(R,SIG_EFF),RT, name='PrincStressState')
        return SIG_EFF_PRI

    def __get_voigt_to_matrix_2d(batch, matrix):
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

        arg = tf.where(cond_num_den_4, tf.atan(tf.divide(num,den)), tf.where(cond_num_den_3, pi_half,tf.zeros_like(num)))
        return arg

    def __get_principal_direction_old(eps):
        with tf.name_scope('Computation_I'):
            with tf.name_scope('Size'):
                batch = tf.cast(tf.shape(eps)[0],tf.int32) # shape of matrix
            with tf.name_scope('Numerator'):
                numerator = eps[:,2]
            with tf.name_scope('Denomimator'):
                denominator = tf.subtract(eps[:,0],eps[:,1])
            with tf.name_scope('Selection'):
                argument = YieldCriteriaUtilities.__case_selection(batch, numerator, denominator)
            with tf.name_scope('Theta_Principal'):
                angle = tf.multiply(tf.constant(0.50),argument, name='Theta_P')
        return angle

    def __get_principal_direction(sig_eff):
        with tf.name_scope('Computation_J'):
            with tf.name_scope('Size'):
                batch = tf.cast(tf.shape(sig_eff)[0],tf.int32) # shape of matrix
            with tf.name_scope('Numerator'):
                numerator = tf.multiply(2.0,sig_eff[:,2])
            with tf.name_scope('Denomimator'):
                denominator = tf.subtract(sig_eff[:,0],sig_eff[:,1])
            with tf.name_scope('Selection'):
                argument = YieldCriteriaUtilities.__case_selection(batch, numerator, denominator)
            with tf.name_scope('Theta_Principal'):
                angle = tf.multiply(tf.constant(0.50),argument, name='Theta_P')
        return angle

    def __get_rotation_matrix(angle):
        with tf.name_scope('RotationMatrix'):
            with tf.name_scope('Computation_K'):
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

    def AlphaRatio(alpha):
        with tf.name_scope('AlphaRatio'):
            ratio = tf.divide(1.0,tf.subtract(1.0,alpha))
        return ratio
# -----------------------------------------------------------------------------
    def TensionCompressionRatio(st, sp):
        with tf.name_scope('TensionCompressionratio'):
            ratio = tf.divide(st, sp)
        return ratio
# -----------------------------------------------------------------------------
    def AuxTerm1(alpha, i1):
        with tf.name_scope('Term_1'):
            term = tf.multiply(alpha,i1)
        return term
# -----------------------------------------------------------------------------
    def AuxTerm2(j2):
        with tf.name_scope('Term_2'):
            term = tf.sqrt(tf.multiply(j2,3.0))
        return term
    def AuxTerm3(s_max, beta):
        with tf.name_scope('Term_3'):
            term = tf.multiply(beta, s_max)
        return term
   

