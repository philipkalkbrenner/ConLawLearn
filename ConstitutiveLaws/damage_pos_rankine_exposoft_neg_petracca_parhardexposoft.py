import tensorflow as tf

from ConLawLearn.YieldCriteria import *
from ConLawLearn.SofteningTypes import *
from ConLawLearn.StressSplitter import *

'''
The class to call the constitutive law for the:
    DAMAGE LAW with:
    Tension:        Rankine Yield Surface
                    Exponential Softening
    Compression:    Petracca Yield Surface
                    Parabolic Hardening & Exponential Softening
'''

class PosRankineExpoSoftNegPetraccaParHardExpoSoft(object):
    def __init__(self, linear_variables, damage_variables):
        self.e    = linear_variables['E']
        self.fcp  = damage_variables['SP']
        self.fcbi = damage_variables['SBI']
        self.gc   = damage_variables['GC']
        self.ft   = damage_variables['FT']
        self.gt   = damage_variables['GT']
        self.r0   = damage_variables['S0']
        self.re   = self.fcp
        self.rp   = damage_variables['SPP']
        
    def GetStress(self, effective_stress):
        '''
            Tension Part
        '''
        with tf.name_scope("PosEquiStress"):
            equivalent_stress_pos = YieldCriterion.Rankine.PositiveEquivalentStress\
                    (effective_stress)
        with tf.name_scope("PosDamVar"):
            damage_pos =   SofteningType.ExponentialSoftening.GetDamageVariable\
                    (equivalent_stress_pos, self.e, self.ft, self.gt)
        with tf.name_scope("PosStressVec"):
            effective_stress_pos = EffectiveStressSplit.GetPositiveStress(effective_stress)
            stress_vector_pos = self.__compute_stress(damage_pos, effective_stress_pos)
            
        '''
            Compression Part
        '''
        with tf.name_scope("NegEquiStress"):           
            equivalent_stress_neg = YieldCriterion.Petracca.NegativeEquivalentStress\
                    (effective_stress, self.r0, self.fcp, self.fcbi, self.ft)
        with tf.name_scope("NegDamVar"):
            damage_neg   = SofteningType.ParabolicHardeningExponentialSoftening.GetDamageVariable\
                    (equivalent_stress_neg, self.e, self.fcp, self.gc, self.r0, self.re, self.rp)
        with tf.name_scope("NegStressVector"):
            effective_stress_neg = EffectiveStressSplit.GetNegativeStress(effective_stress)
            stress_vector_neg = self.__compute_stress(damage_neg, effective_stress_neg)

        '''
            Total Part
        '''
        with tf.name_scope("TotalStressVec"):
            stress_vector = tf.add(stress_vector_pos, stress_vector_neg)

        return stress_vector



    '''
    Internal Functions
    '''

    def __compute_stress(self, damage, effective_stress):
        with tf.name_scope("Apply1MinusD"):
            stress_vector = tf.multiply(tf.subtract(1.0, tf.expand_dims(damage,1)), effective_stress)
        return stress_vector
