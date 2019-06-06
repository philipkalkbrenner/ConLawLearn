import tensorflow as tf

from ConLawLearn.SofteningTypes.linear_softening import *
from ConLawLearn.SofteningTypes.exponential_softening import *
from ConLawLearn.SofteningTypes.parabolic_hardening_exponential_softening import *
from ConLawLearn.SofteningTypes.bezier_hardening_softening import *

class SofteningType(object):

    class LinearSoftening(object):
        def GetDamageThreshold(equivalent_stress, yield_stress):
            damage_threshold = GetDamageThresholdLinearSoftening(equivalent_stress, yield_stress)
            return damage_threshold
        
        def GetDamageVariable(equivalent_stress, yield_stress, bounding_damage_threshold):
            damage_variable = GetDamageVariableLinearSoftening(equivalent_stress, yield_stress, bounding_damage_threshold)
            return damage_variable

    class ExponentialSoftening(object):

        def GetDamageThreshold(equivalent_stress, yield_stress):
            damage_threshold = GetDamageThresholdExponentialSoftening(equivalent_stress, yield_stress)
            return damage_threshold
        
        def GetDamageVariable(equivalent_stress, young_modulus, yield_stress, fracture_energy):
            damage_variable = GetDamageVariableExponentialSoftening(equivalent_stress, young_modulus, yield_stress, fracture_energy)
            return damage_variable

    class ParabolicHardeningExponentialSoftening(object):

        def GetDamageThreshold(equivalent_stress, damage_onset_stress):
            damage_threshold = GetDamageThresholdParabolicHardeningExponentialSoftening(\
                               equivalent_stress, damage_onset_stress)
            return damage_threshold
        
        def GetDamageVariable(equivalent_stress, young_modulus, yield_stress, \
                              fracture_energy, damage_onset_threshold, \
                              yield_threshold, bounding_damage_threshold):
            damage_variable = GetDamageVariableParabolicHardeningExponentialSoftening(\
                              equivalent_stress, young_modulus, yield_stress, fracture_energy, \
                              damage_onset_threshold, yield_threshold, bounding_damage_threshold)
            return damage_variable 

    class BezierHardeningSoftening(object):

        def GetDamageThreshold(equivalent_stress, damage_onset_stress):
            damage_threshold = GetDamageThresholdBezierHardeningSoftening(equivalent_stress, damage_onset_stress)
            return damage_threshold

        def GetDamageVariable(tau, e, e0, ei, ep, ej, ek, er, eu, \
                              s0, si, sp, sj, sk, sr, su):
            damage_variable = GetDamageVariableBezierHardeningSoftening(\
                                tau, e, e0, ei, ep, ej, ek, er, eu, \
                                s0, si, sp, sj, sk, sr, su)
            return damage_variable   
             
    '''
-------------------------------------------------------------------------------
    '''
