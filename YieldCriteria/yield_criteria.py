import tensorflow as tf
import numpy as np
import sys

from ConLawLearn.YieldCriteria.petracca_yield_criteria import *
from ConLawLearn.YieldCriteria.lubliner_yield_criteria import *
from ConLawLearn.YieldCriteria.drucker_prager_yield_criteria import *
from ConLawLearn.YieldCriteria.rankine_yield_criteria import *

# -----------------------------------------------------------------------------
class YieldCriterion(object):
    
    class DruckerPrager(object):
        ''' Drucker Prager Yield Criterion
            k1 = 0.0
        '''
        def NegativeEquivalentStress(SIG_EFF, fc0, fcp, fcbi, ft):
            tau = NegativeEquivalentStressDruckerPrager(SIG_EFF, fc0, fcp, fcbi, ft)
            return tau

        def PositiveEquivalentStress(SIG_EFF, fcp, fcbi, ft):
            YieldCriterion._Error_Message_Tension_Yield("Drucker Prager")
    # -------------------------------------------------------------------------

    class Lubliner(object):
        ''' Lubliner Yield Criterion
            k1 = 1.0
        '''
        def NegativeEquivalentStress(SIG_EFF, fc0, fcp, fcbi, ft):
            tau = NegativeEquivalentStressLubliner(SIG_EFF, fc0, fcp, fcbi, ft)
            return tau

        def PositiveEquivalentStress(SIG_EFF, fcp, fcbi, ft):
            YieldCriterion._Error_Message_Tension_Yield("Lubliner")
            
    # -------------------------------------------------------------------------

    class Petracca(object):
        ''' 
            Petracca Modified Yield Criterion
            k1 = 0.16
        '''
        def NegativeEquivalentStress(SIG_EFF, fc0, fcp, fcbi, ft):
            tau = NegativeEquivalentStressPetracca(SIG_EFF, fc0, fcp, fcbi, ft)
            return tau

        def PositiveEquivalentStress(SIG_EFF, fcp, fcbi, ft):
            tau = PositiveEquivalentStressPetracca(SIG_EFF, fcp, fcbi, ft)
            return tau
    # -------------------------------------------------------------------------

    class Rankine(object):
        ''' 
            Rankine Yield Criteria
        '''
        def NegativeEquivalentStress():
            negative_equivalent_stress = NegativeEquivalentStressRankine(SIG_EFF, fc0)
            return negative_equivalent_stress

        def PositiveEquivalentStress():
            positive_equivalent_stress = PositiveEquivalentStressRankine(SIG_EFF, ft)
            return positive_equivalent_stress

    # -------------------------------------------------------------------------



    def _Error_Message_Tension_Yield(name):
        print(name, "Yield criteria is not implemented for tension")
        print(" --> please choose for the TENSION Yield type one of the following:")
        print("        - Rankine")
        print("        - Petracca")
        sys.exit()

'''
    def SMAX(sig_eff):
        s1,s2 = YieldCriterion._s1_s2(sig_eff)
        smax = YieldCriterion._Smax(s1,s2)
        return smax
    def S1(sig_eff):
        s1,s2 = YieldCriterion._s1_s2(sig_eff)
        return s1
    def S2(sig_eff):
        s1,s2 = YieldCriterion._s1_s2(sig_eff)
        return s2
    def SMIN(sig_eff):
        s1,s2 = YieldCriterion._s1_s2(sig_eff)
        smin = YieldCriterion._Smin(s1,s2)
        return smin

    
    def _s1_s2(sig_eff):
        s1,s2 = YieldCriterion.__get_principal_stress_values(sig_eff)
        return s1,s2
    def _Smax(s1,s2):
        smax = YieldCriterion.__get_equivalent_stress_smax(s1, s2)
        return smax

    def _Smin(s1,s2):
        smin = YieldCriterion.__get_equivalent_stress_smin(s1, s2)
        return smin
    
    def _I1(eps,e,nu):
        i1 = YieldCriterion.__get_first_invariant(eps, e, nu)
        return i1

    def _J2(eps,e,nu):
        j2 = YieldCriterion.__get_second_deviatoric_invariant(eps, e, nu)
        return j2
    def Alpha(fcbi, fcp):
        kb = YieldCriterion.__get_kb(fcbi, fcp)
        alpha = YieldCriterion.__get_alpha(kb)
        return alpha

    def GetDirection(sig_eff, eps):
        T = YieldCriterion.__get_principal_direction(sig_eff)
        T1 = YieldCriterion.__get_principal_direction_old(eps)
        return (T,T1)

        
    def _Negative_Equivalent_Stress(SIG_EFF, fc0, fcp, fcbi, ft, k1):
        kb      = YieldCriterion.__get_kb(fcbi, fcp)
        alpha   = YieldCriterion.__get_alpha(kb)
        beta    = YieldCriterion.__get_beta(fc0, ft, alpha)
        i1      = YieldCriterion.__get_first_invariant(SIG_EFF)
        j2      = YieldCriterion.__get_second_deviatoric_invariant(SIG_EFF)
        s1, s2  = YieldCriterion.__get_principal_stress_values(SIG_EFF)
        s_max   = YieldCriterion.__get_equivalent_stress_smax(s1, s2)
        s_min_heavi = YieldCriterion.__get_equivalent_stress_smin_heavi(s1,s2)

        ratio1  = YieldCriterion.__get_equivalent_stress_ratio1(alpha)
        term1   = YieldCriterion.__get_equivalent_stress_term1(alpha, i1)
        term2   = YieldCriterion.__get_equivalent_stress_term2(j2)
        term3   = YieldCriterion.__get_equivalent_stress_term3(s_max, beta)

        tau = tf.multiply(tf.multiply(ratio1, tf.add(tf.add(term1, term2),\
                                         tf.multiply(k1,term3))),s_min_heavi)
        return tau

    def _Positive_Equivalent_Stress(SIG_EFF, fcp, fcbi, ft):
        kb      = YieldCriterion.__get_kb(fcbi, fcp)
        alpha   = YieldCriterion.__get_alpha(kb)
        beta    = YieldCriterion.__get_beta(fcp, ft, alpha)
        i1      = YieldCriterion.__get_first_invariant(SIG_EFF)
        j2      = YieldCriterion.__get_second_deviatoric_invariant(SIG_EFF)
        s1, s2  = YieldCriterion.__get_principal_stress_values(SIG_EFF)
        s_max   = YieldCriterion.__get_equivalent_stress_smax(s1, s2)
        s_max_heavi = YieldCriterion.__get_equivalent_stress_smax_heavi(s1,s2)

        ratio1  = YieldCriterion.__get_equivalent_stress_ratio1(alpha)
        ratio2  = YieldCriterion.__get_equivalent_stress_ratio2(ft, fcp)
        term1   = YieldCriterion.__get_equivalent_stress_term1(alpha, i1)
        term2   = YieldCriterion.__get_equivalent_stress_term2(j2)
        term3   = YieldCriterion.__get_equivalent_stress_term3(s_max, beta)

        tau = tf.multiply(tf.multiply(tf.multiply(ratio1, tf.add(term1, tf.add(term2,term3))) \
                      ,ratio2), s_max_heavi)
        return tau

    # ----------------------------------------------------------------------------

    def __get_kb(fcbi, fcp):
        kb = tf.divide(fcbi,fcp, name="kb")
        return kb
    
    def __get_alpha(kb):
        alpha = tf.divide(tf.subtract(kb, 1.0), \
                          tf.subtract(tf.multiply(2.0, kb), 1), name= "alpha")
        return alpha
    
    def __get_beta(fcp, ft, alpha):
        b_temp1 = tf.multiply(tf.divide(fcp, ft),tf.subtract(1.0,alpha))
        b_temp2 = tf.add(1.0,alpha)
        beta = tf.subtract(b_temp1, b_temp2, name="beta")
        return beta

    def __get_first_invariant_old(epsilon, e, nu):
        with tf.name_scope('1stInvariant'):
            batch = tf.cast(tf.shape(epsilon)[0],tf.int32)
            with tf.name_scope('Computation_G'):
                fac_i1 = tf.add(epsilon[:,0], epsilon[:,1])
                i1_temp = tf.divide(tf.multiply(tf.multiply(-1.0, fac_i1), e),\
                                    tf.subtract(nu, 1.0))
                condition_i1 = tf.equal(fac_i1,0.0)
            i1 = tf.where(condition_i1, tf.zeros([batch]),i1_temp, name="I1")
        return i1

    def __get_first_invariant(sig_eff):
        with tf.name_scope('1stInvariant'):
            i1 = tf.add(sig_eff[:,0], sig_eff[:,1], name="I1")
        return i1

    def __get_second_deviatoric_invariant(sig_eff):
        with tf.name_scope('2ndDevInvariant'):
            batch = tf.cast(tf.shape(sig_eff)[0],tf.int32)
            with tf.name_scope('Computation_H'):
                j2_sum1 = tf.pow(sig_eff[:,0],2, name = "j2_sum1")
                j2_sum2 = tf.multiply(sig_eff[:,0], sig_eff[:,1], name = "j2_sum2")
                j2_sum3 = tf.pow(sig_eff[:,1],2, name = "j2_sum3")
                j2_sum4 = tf.multiply(3.0,tf.pow(sig_eff[:,2],2), name="j2_sum4")

                j2 = tf.divide(tf.subtract(tf.add(tf.add(j2_sum1, j2_sum3),\
                                        j2_sum4),j2_sum2),3.0, name="J2")
        #        condition_j2 = tf.equal(j2, 0.0)
        #    J2 = tf.where(condition_j2, tf.zeros([batch]), j2_temp)
        #return J2
        return j2

    def __get_principal_stress_values(sig_eff):
        sig_eff_pri = YieldCriterion.__get_principal_stress(sig_eff)
        with tf.name_scope('PrincStresses'):
            with tf.name_scope('Sigma1'):
                s1 = sig_eff_pri[:,0][:,0]
            with tf.name_scope('Sigma2'):
                s2 = sig_eff_pri[:,1][:,1]
        return s1, s2
    
    def __get_equivalent_stress_smax(s1, s2):
        with tf.name_scope('S_max'):
            zeros = tf.zeros(tf.shape(s1))
            s_max = tf.where(tf.greater(s1, s2), s1, s2)
            s_max_macaulay = tf.where(tf.greater_equal(s_max, zeros), s_max, zeros, name = "s_max_macaulay")
        return s_max_macaulay

    def __get_equivalent_stress_smax_heavi(s1,s2):
        zeros = tf.zeros(tf.shape(s1))
        ones  = tf.add(zeros, 1.0)
        s_max = tf.where(tf.greater(s1, s2), s1, s2)
        smax_heavi = tf.where(tf.greater(s_max,zeros), ones, zeros)
        return smax_heavi

    def __get_equivalent_stress_smin(s1, s2):
        with tf.name_scope('S_min'):
            zeros = tf.zeros(tf.shape(s1))
            s_min = tf.where(tf.less(s1, s2), s1, s2)
        return s_min

    def __get_equivalent_stress_smin_heavi(s1,s2):
        zeros = tf.zeros(tf.shape(s1))
        ones  = tf.add(zeros, 1.0)
        s_min = tf.where(tf.less(s1, s2), s1, s2)
        s_min_heavi = tf.where(tf.greater(s_min,zeros), zeros, ones)
        return s_min_heavi

    def __get_principal_stress(sig_eff):
        batch = tf.cast(tf.shape(sig_eff)[0],tf.int32)
        SIG_EFF = YieldCriterion.__get_voigt_to_matrix_2d(batch, sig_eff)
    
        THETA = YieldCriterion.__get_principal_direction(sig_eff)
        R,RT  = YieldCriterion.__get_rotation_matrix(THETA)
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
        cond_num = tf.less(tf.abs(num),1e-12, name='CondNumerator')
        cond_den = tf.less(tf.abs(den),1e-12, name='CondDenominator')
        cond_num_den = tf.logical_and(tf.less(tf.abs(num),1e-12), tf.less(tf.abs(den),1e-12))
        pi_half  = tf.fill([batch],tf.divide(np.pi,2.0), name='PiHalf')

        den = tf.where(cond_num_den, tf.ones_like(den), den)
        inner_where = tf.where(cond_den, pi_half, tf.atan(tf.divide(num,den)), \
                            name='Inner_Request')
        arg = tf.where(cond_num,tf.zeros([batch]),inner_where, \
                        name='TanArgu')
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
                argument = YieldCriterion.__case_selection(batch, numerator, denominator)
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
                argument = YieldCriterion.__case_selection(batch, numerator, denominator)
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

    def __get_equivalent_stress_ratio1(alpha):
        with tf.name_scope('Ratio_1'):
            ratio = tf.divide(1.0,tf.subtract(1.0,alpha))
        return ratio
# -----------------------------------------------------------------------------
    def __get_equivalent_stress_ratio2(st, sp):
        with tf.name_scope('Ratio_2'):
            ratio = tf.divide(st, sp)
        return ratio
# -----------------------------------------------------------------------------
    def __get_equivalent_stress_term1(alpha, i1):
        with tf.name_scope('Term_1'):
            term = tf.multiply(alpha,i1)
        return term
# -----------------------------------------------------------------------------
    def __get_equivalent_stress_term2(j2):
        with tf.name_scope('Term_2'):
            term = tf.sqrt(tf.multiply(j2,3.0))
        return term
    def __get_equivalent_stress_term3(s_max, beta):
        with tf.name_scope('Term_3'):
            term = tf.multiply(beta, s_max)
        return term
   

'''
