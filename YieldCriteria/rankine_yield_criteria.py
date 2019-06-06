import tensorflow as tf
import numpy as np

from ConLawLearn.YieldCriteria.yield_criteria_utilities import YieldCriteriaUtilities

def NegativeEquivalentRankine(SIG_EFF):
    with tf.name_scope("RankineYieldNegative"):
        principal_stress_1, principal_stress_2  = \
                YieldCriteriaUtilities.PrincipalStressValues(SIG_EFF)
        minimum_pricipal_stress = \
                YieldCriteriaUtilities.PrincipalStressMin(principal_stress_1, principal_stress_2)
    return minimum_pricipal_stress

def PositiveEquivalentRankine(SIG_EFF):
    with tf.name_scope("RankineYieldPositive"):
        principal_stress_1, principal_stress_2  = \
                YieldCriteriaUtilities.PrincipalStressValues(SIG_EFF)
        maximum_pricipal_stress = \
                YieldCriteriaUtilities.PrincipalStressMax(principal_stress_1, principal_stress_2)
    return maximum_principal_stress

