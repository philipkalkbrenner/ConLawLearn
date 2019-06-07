import tensorflow as tf

from ConLawLearn.YieldCriteria import *
from ConLawLearn.SofteningTypes import *
from ConLawLearn.StressSplitter import *

'''
The class to call the constitutive law for the:
    DAMAGE LAW with:
    Tension:        Petracca Yield Surface
                    Linear Softening
    Compression:    Drucker Prager Yield Surface
                    Linear Softening
'''

class PosPetraccaLinSoftNegDrucPragLinSoft(object):
    def __init__(self, linear_variables, damage_variables):
        self.e    = linear_variables['E']
        self.fcp  = damage_variables['FCP']
        self.fcpb  = damage_variables['FCPB']
        self.fcbi = damage_variables['FCBI']
        self.ft   = damage_variables['FT']
        self.ftb   = damage_variables['FTB']
        
    def GetStress(self, effective_stress):
        '''
            Tension Part
        '''
        with tf.name_scope("PosEquiStress"):
            equivalent_stress_pos = YieldCriterion.Petracca.PositiveEquivalentStress\
                    (effective_stress, self.fcp, self.fcbi, self.ft)
        with tf.name_scope("PosDamVar"):
            damage_pos =   SofteningType.LinearSoftening.GetDamageVariable\
                    (equivalent_stress_pos, self.ft, self.ftb)
        with tf.name_scope("PosStressVec"):
            effective_stress_pos = EffectiveStressSplit.GetPositiveStress(effective_stress)
            stress_vector_pos = self.__compute_stress(damage_pos, effective_stress_pos)
            
        '''
            Compression Part
        '''
        with tf.name_scope("NegEquiStress"):           
            equivalent_stress_neg = YieldCriterion.DruckerPrager.NegativeEquivalentStress\
                    (effective_stress, self.fcp, self.fcp, self.fcbi, self.ft)
        with tf.name_scope("NegDamVar"):
            damage_neg   = SofteningType.LinearSoftening.GetDamageVariable\
                    (equivalent_stress_neg, self.fcp, self.fcpb)
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
