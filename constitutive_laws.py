import tensorflow as tf

from ConLawLearn.yield_criterions import YieldCriterion
from ConLawLearn.softening_types import SofteningType
from ConLawLearn.splitting_stress import EffectiveStressSplit

'''
AVAILABLE CONSTITUTIVE LAWS:

    A) Linear Elastic Law
              A.1)   LinearElastic.GetStress(...) 
    B) Drucker Prager Yield Surface:
        B.1) Tension:     Exponential Softening
             Compression: Exponential Softening
              B.1.1) PosDrucPragExpoSoftNegDrucPragExpoSoft(...).GetStress()
              B.1.2) PureCompressionDrucPragExpoSoft.GetStress(...)
              B.1.3) PureTensionDrucPragExpoSoft.GetStress(...)
                     
        B.2) Tension:      Exponential Softening
             Compression:  Parabolic Hardening and Exponential Softening
              B.2.1) PosDrucPragExpoSoftNegDrucPragParHardExpoSoft(...).GetStress()
              B.2.2) PureCompressionDrucPragParHardExpoSoft.GetStress(...)
              B.2.3) PureTensionDrucPragExpoSoft.GetStress(...)
                     
        B.3) Tension:      Exponential Softening
             Compression:  Bezier Curve Hardening and Softening
              B.3.1) PosDrucPragExpoSoftNegDrucPragBezierHardSoft(...).GetStress()
              B.3.2) PureCompressionDrucPragBezierHardSoft.GetStress(...)
              B.3.3) PureTensionDrucPragExpoSoft.GetStress(...)
              
    C) Lubliner Yield Surface:
        C.1) Tension:     Exponential Softening
             Compression: Exponential Softening
              C.1.1) PosLublinerExpoSoftNegLublinerExpoSoft(...).GetStress()
              C.1.2) PureCompressionLublinerExpoSoft.GetStress(...)
              C.1.3) PureTensionLublinerExpoSoft.GetStress(...)
                
        C.2) Tension:      Exponential Softening
             Compression:  Parabolic Hardening and Exponential Softening
              C.2.1) PosLublinerExpoSoftNegLublinerParHardExpoSoft(...).GetStress()
              C.2.2) PureCompressionLublinerParHardExpoSoft.GetStress(...)
              C.2.3) PureTensionLublinerExpoSoft.GetStress(...)
              
        C.3) Tension:      Exponential Softening
             Compression:  Bezier Curve Hardening and Softening
              C.3.1) PosLublinerExpoSoftNegLublinerBezierHardSoft(...).GetStress()
              C.3.2) PureCompressionLublinerBezierHardSoft.GetStress(...) 
              C.3.3) PureTensionLublinerExpoSoft.GetStress(...)
                
    D) Petracca Modified Yield Surface:
        D.1) Tension:     Exponential Softening
             Compression: Exponential Softening
              D.1.1) PosPetraccaExpoSoftNegPetraccaExpoSoft(...).GetStress()
              D.1.2) PureCompressionPetraccaExpoSoft.GetStress(...)
              D.1.3) PureTensionPetraccaExpoSoft.GetStress(...)
        D.2) Tension:      Exponential Softening
             Compression:  Parabolic Hardening and Exponential Softening
              D.2.1) PosPetraccaExpoSoftNegPetraccaParaHardExpoSoft(...).GetStress()
              D.2.2) PureCompressionPetraccaParHardExpoSoft.GetStress(...)
              D.2.3) PureTensionPetraccaExpoSoft.GetStress(...)
        D.3) Tension:      Exponential Softening
             Compression:  Bezier Curve Hardening and Softening
              D.3.1) PosPetraccaExpoSoftNegPetraccaBezierHardSoft(...).GetStress() 
              D.3.2) PureCompressionPetraccaBezierHardSoft.GetStress(...) 
              D.3.3) PureTensionPetraccaExpoSoft.GetStress(...)   
'''

class ConstitutiveLaw(object):
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
    Type A:
    Linear Elastic Contitutive Law
    '''

    class LinearElasticPlaneStress(object):
        def GetEffectiveStress(eps, variables):
            with tf.name_scope("LinearElasticLawPlaneStress"):
               nu = variables['NU']
               e = variables['E']
               with tf.name_scope("ElasticityTensor"):
                  C = ConstitutiveLaw._Elasticity_Tensor_Plane_Stress(nu, e)
               with tf.name_scope("EffectiveStressVector"):
                  SIG_EFF = tf.matmul(eps, C)
            return SIG_EFF
        def GetLinearElasticStress(eps, variables):
            with tf.name_scope("LinearElasticLawPlaneStress"):
               nu = variables['NU']
               e = variables['E']
               with tf.name_scope("ElasticityTensor"):
                  C = ConstitutiveLaw._Elasticity_Tensor_Plane_Stress(nu, e)
               with tf.name_scope("LinearElasticStressVector"):
                  SIG = tf.matmul(eps, C)
            return SIG
        def GetStress(eps, variables):
            with tf.name_scope("LinearElasticLawPlaneStress"):
               nu = variables['NU']
               e = variables['E']
               with tf.name_scope("ElasticityTensor"):
                  C = ConstitutiveLaw._Elasticity_Tensor_Plane_Stress(nu, e)
               with tf.name_scope("LinearElasticStressVector"):
                  SIG = tf.matmul(eps, C)
            return SIG
        
    class LinearElasticPlaneStrain(object):
        def GetEffectiveStress(eps, variables):
            with tf.name_scope("LinearElasticLawPlaneStrain"):
                nu = variables['NU']
                e = variables['E']
                with tf.name_scope("ElasticityTensor"):
                    C = ConstitutiveLaw._Elasticity_Tensor_Plane_Strain(nu, e)
                with tf.name_scope("EffectiveStressVector"):
                    SIG_EFF = tf.matmul(eps, C)
            return SIG_EFF
        def GetLinearElasticStress(eps, variables):
            with tf.name_scope("LinearElasticLawPlaneStrain"):
                nu = variables['NU']
                e = variables['E']
                with tf.name_scope("ElasticityTensor"):
                    C = ConstitutiveLaw._Elasticity_Tensor_Plane_Strain(nu, e)
                with tf.name_scope("EffectiveStressVector"):
                    SIG = tf.matmul(eps, C)
            return SIG
        def GetStress(eps, variables):
            with tf.name_scope("LinearElasticLawPlaneStrain"):
                nu = variables['NU']
                e = variables['E']
                with tf.name_scope("ElasticityTensor"):
                    C = ConstitutiveLaw._Elasticity_Tensor_Plane_Strain(nu, e)
                with tf.name_scope("EffectiveStressVector"):
                    SIG = tf.matmul(eps, C)
            return SIG
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
    Type B:
    All Classes with Drucker Prager Yield
    '''
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type B.1:
    '''
# -----------------------------------------------------------------------------
    '''
       Type B.1.1:
    '''
    class PosDrucPragExpoSoftNegDrucPragExpoSoft(object):
        '''
           Tension:        Drucker Prager Yield Surface
                           Exponential Softening
           Compression:    Drucker Prager Yield Surface
                           Exponential Softening
        '''
        def __init__(self, sig_eff, vars_le, vars_nl):
            e = vars_le['E']
            fcp = vars_nl['SP']
            fcbi = vars_nl['SBI']
            gc = vars_nl['GC']
            ft = vars_nl['FT']
            gt = vars_nl['GT']
            '''
               Tension Part
            '''
            with tf.name_scope("PosEquiStress"):
                tau_pos = YieldCriterion.DruckerPrager.PositiveEquivalentStress\
                      (sig_eff, fcp, fcbi, ft)
            with tf.name_scope("PosDamVar"):
                d_pos =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_pos, e, ft, gt)
            with tf.name_scope("PosStressVec"):
                sig_eff_pos = EffectiveStressSplit.GetPositiveStress(sig_eff) 
                sig_pos = ConstitutiveLaw._Compute_Stress(d_pos, sig_eff_pos)

            '''
               Compression Part
            '''
            with tf.name_scope("NegEquiStress"):            
                tau_neg = YieldCriterion.DruckerPrager.NegativeEquivalentStress\
                      (sig_eff, fcp, fcp, fcbi, ft)
            with tf.name_scope("NegDamVar"):
                d_neg =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_neg, e, fcp, gc)
            with tf.name_scope("PosStressVec"):
                sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
                sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)

            '''
               Total Part
            '''
            with tf.name_scope("TotalStressVec"):
                sig = tf.add(sig_pos, sig_neg)

            # -----------------------------------------------------------------
            # Caller 
            # -----------------------------------------------------------------
            self.GetStress = sig
            # -----------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type B.1.2:
    '''
    class PureCompressionDrucPragExpoSoft(object):
        '''
           Tension:        Not considered in this case, since input strains and
                           stresses are only in pure compression state
           Compression:    Drucker Prager Yield Surface
                           Exponential Softening
        '''
        def GetStress(sig_eff, vars_le, vars_nl):
            e = vars_le['E']
            fcp = vars_nl['SP']
            fcbi = vars_nl['SBI']
            gc = vars_nl['GC']
            ft = vars_nl['FT']
            
            '''
               Compression Part
            '''            
            tau_neg = YieldCriterion.DruckerPrager.NegativeEquivalentStress\
                      (sig_eff, fcp, fcp, fcbi, ft)
            d_neg =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_neg, e, fcp, gc)
            sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
            sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)
            
            return sig_neg
# -----------------------------------------------------------------------------
    '''
       Type B.1.3:
    '''
    class PureTensionDrucPragExpoSoft(object):
        '''
           Tension:        Drucker Prager Yield Surface
                           Exponential Softening
           Compression:    Not considered in this case, since input strains and
                           stresses are only in pure tension state
        '''
        def GetStress(sig_eff, vars_le, vars_nl):
            e = vars_le['E']
            fcp = vars_nl['SP']
            fcbi = vars_nl['SBI']
            ft = vars_nl['FT']
            gt = vars_nl['GT']
            
            '''
               Tension Part
            '''
            tau_pos = YieldCriterion.DruckerPrager.PositiveEquivalentStress\
                      (sig_eff, fcp, fcbi, ft)
            d_pos =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_pos, e, ft, gt)
            sig_eff_pos = EffectiveStressSplit.GetPositiveStress(sig_eff)
            sig_pos = ConstitutiveLaw._Compute_Stress(d_pos, sig_eff_pos)

            return sig_pos
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type B.2:
    '''
# -----------------------------------------------------------------------------
    '''
       Type B.2.1:
    '''
    class PosDrucPragExpoSoftNegDrucPragParHardExpoSoft(object):
        '''
           Tension:        Drucker Prager Yield Surface
                           Exponential Softening
           Compression:    Drucker Prager Yield Surface
                           Parabolic Hardening & Exponential Softening
        '''
        def __init__(self, sig_eff, vars_le, vars_nl):
            e    = vars_le['E']
            fcp  = vars_nl['SP']
            fcbi = vars_nl['SBI']
            gc   = vars_nl['GC']
            ft   = vars_nl['FT']
            gt   = vars_nl['GT']
            r0   = vars_nl['S0']
            re   = fcp
            rp   = vars_nl['SPP']

            '''
               Tension Part
            '''
            with tf.name_scope("PosEquiStress"):
                tau_pos = YieldCriterion.DruckerPrager.PositiveEquivalentStress\
                      (sig_eff, fcp, fcbi, ft)
            with tf.name_scope("PosDamVar"):
                d_pos =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_pos, e, ft, gt)
            with tf.name_scope("PosStressVec"):
                sig_eff_pos = EffectiveStressSplit.GetPositiveStress(sig_eff)
                sig_pos = ConstitutiveLaw._Compute_Stress(d_pos, sig_eff_pos)
            
            '''
               Compression Part
            '''            
            with tf.name_scope("NegEquiStress"):
                tau_neg = YieldCriterion.DruckerPrager.NegativeEquivalentStress\
                      (sig_eff, r0, fcp, fcbi, ft)
            with tf.name_scope("NegDamVar"):
                d_neg   = SofteningType.ParabolicHardeningExponentialSoftening.\
                      GetDamageVariable(tau_neg, e, fcp, gc, r0, re, rp)
            with tf.name_scope("NegStressVec"):
                sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
                sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)
            
            '''
               Total Part
            '''
            with tf.name_scope("TotalStressVec"):
                 sig = tf.add(sig_pos, sig_neg)

            # -----------------------------------------------------------------
            # Caller ----------------------------------------------------------
            self.GetStress = sig
            # -----------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type B.2.2:
    '''
    class PureCompressionDrucPragParHardExpoSoft(object):
        '''
           Tension:        Not considered in this case, since input strains and
                           stresses are only in pure compression state
           Compression:    Drucker Prager Yield Surface
                           Parabolic Hardening & Exponential Softening
        '''
        def GetStress(sig_eff, vars_le, vars_nl):
            e    = vars_le['E']
            fcp  = vars_nl['SP']
            fcbi = vars_nl['SBI']
            gc   = vars_nl['GC']
            ft   = vars_nl['FT']
            gt   = vars_nl['GT']
            r0   = vars_nl['S0']
            re   = fcp
            rp   = vars_nl['SPP']
            
            '''
               Compression Part
            '''            
            tau_neg = YieldCriterion.DruckerPrager.NegativeEquivalentStress\
                      (sig_eff, r0, fcp, fcbi, ft)
            d_neg   = SofteningType.ParabolicHardeningExponentialSoftening.\
                      GetDamageVariable(tau_neg, e, fcp, gc, r0, re, rp)
            sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
            sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)
            
            return sig_neg
# -----------------------------------------------------------------------------
    '''
       Type B.2.3  = Type B.1.3
    '''
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type B.3:
    '''
# -----------------------------------------------------------------------------
    '''
       Type B.3.1:
    '''
    class PosDrucPragExpoSoftNegDrucPragBezierHardSoft(object):
        '''
           Tension:        Drucker Prager Yield Surface
                           Exponential Softening
           Compression:    Drucker Prager Yield Surface
                           Bezier Hardening & Softening
        '''
        def __init__(self, sig_eff, vars_le, vars_nl):
            e = vars_le['E']
            e0 = vars_nl['E0']
            ei = vars_nl['EI']
            ep = vars_nl['EP']
            ej = vars_nl['EJ']
            ek = vars_nl['EK']
            er = vars_nl['ER']
            eu = vars_nl['EU']
            s0 = vars_nl['S0']
            si = vars_nl['SI']
            sp = vars_nl['SP']
            sj = vars_nl['SJ']
            sk = vars_nl['SK']
            sr = vars_nl['SR']
            su = vars_nl['SU']
            fcbi = vars_nl['SBI']
            ft = vars_nl['FT']
            gt = vars_nl['GT']

            '''
               Tension Part
            '''
            with tf.name_scope("PosEquiStress"):
                tau_pos = YieldCriterion.DruckerPrager.PositiveEquivalentStress\
                      (sig_eff, sp, fcbi, ft)
            with tf.name_scope("PosDamVar"):
                d_pos =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_pos, e, ft, gt)
            with tf.name_scope("PosStressVec"):
                sig_eff_pos = EffectiveStressSplit.GetPositiveStress(sig_eff)
                sig_pos = ConstitutiveLaw._Compute_Stress(d_pos, sig_eff_pos)
            
            '''
               Compression Part
            '''            
            with tf.name_scope("NegEquiStress"):
                tau_neg = YieldCriterion.DruckerPrager.NegativeEquivalentStress\
                      (sig_eff, s0, sp, fcbi, ft)
            with tf.name_scope("NegDamVar"):
                d_neg   = SofteningType.BezierHardeningSoftening.GetDamageVariable\
                      (tau_neg, e, e0, ei, ep, ej, ek, er, eu, \
                       s0, si, sp, sj, sk, sr, su)
            with tf.name_scope("NegStressVec"):
                sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
                sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)
            
            '''
               Total Part
            '''
            with tf.name_scope("TotalStressVec"):
                sig = tf.add(sig_pos, sig_neg)

            # -----------------------------------------------------------------
            # Caller ---------------------------------------------------------- 
            self.GetStress = sig
            # -----------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type B.3.2:
    '''
    class PureCompressionDrucPragBezierHardSoft(object):
        '''
           Tension:        Not considered in this case, since input strains and
                           stresses are only in pure compression state
           Compression:    Drucker Prager Yield Surface
                           Bezier Hardening & Softening
        '''
        def GetStress(sig_eff, vars_le, vars_nl):
            e = vars_le['E']
            e0 = vars_nl['E0']
            ei = vars_nl['EI']
            ep = vars_nl['EP']
            ej = vars_nl['EJ']
            ek = vars_nl['EK']
            er = vars_nl['ER']
            eu = vars_nl['EU']
            s0 = vars_nl['S0']
            si = vars_nl['SI']
            sp = vars_nl['SP']
            sj = vars_nl['SJ']
            sk = vars_nl['SK']
            sr = vars_nl['SR']
            su = vars_nl['SU']
            fcbi = vars_nl['SBI']
            ft = vars_nl['FT']
            gt = vars_nl['GT']
            '''
               Compression Part
            '''            
            tau_neg = YieldCriterion.DruckerPrager.NegativeEquivalentStress\
                      (sig_eff, s0, sp, fcbi, ft)
            d_neg   = SofteningType.BezierHardeningSoftening.GetDamageVariable\
                      (tau_neg, e, e0, ei, ep, ej, ek, er, eu, \
                       s0, si, sp, sj, sk, sr, su)
            sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
            sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)
            
            return sig_neg
# -----------------------------------------------------------------------------
    '''
       Type B.3.3  = Type B.1.3
    '''
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
        Type C:
        All Classes with Lubliner Yield
    '''
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type C.1:
    '''
# -----------------------------------------------------------------------------
    '''
       Type C.1.1:
    '''
    class PosLublinerExpoSoftNegLublinerExpoSoft(object):
        '''
           Tension:        Lubliner Yield Surface
                           Exponential Softening
           Compression:    Lubliner Yield Surface
                           Exponential Softening
        '''
        def __init__(self, sig_eff, vars_le, vars_nl):
            e = vars_le['E']
            fcp = vars_nl['SP']
            fcbi = vars_nl['SBI']
            gc = vars_nl['GC']
            ft = vars_nl['FT']
            gt = vars_nl['GT']

            '''
               Tension Part
            '''
            with tf.name_scope("PosEquiStress"): 
                tau_pos = YieldCriterion.Lubliner.PositiveEquivalentStress\
                      (sig_eff, fcp, fcbi, ft)
            with tf.name_scope("PosDamVar"):
                d_pos =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_pos, e, ft, gt)
            with tf.name_scope("PosStressVec"):
                sig_eff_pos = EffectiveStressSplit.GetPositiveStress(sig_eff)
                sig_pos = ConstitutiveLaw._Compute_Stress(d_pos, sig_eff_pos)
            
            '''
               Compression Part
            '''            
            with tf.name_scope("NegEquiStress"):
                tau_neg = YieldCriterion.Lubliner.NegativeEquivalentStress\
                      (sig_eff, fcp, fcp, fcbi, ft)
            with tf.name_scope("NegDamVar"):
                d_neg =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_neg, e, fcp, gc)
            with tf.name_scope("NegStressVec"):
                sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
                sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)
            
            '''
               Total Part
            '''
            with tf.name_scope("TotalStressVec"):
                sig = tf.add(sig_pos, sig_neg)

            # -----------------------------------------------------------------
            # Caller ----------------------------------------------------------
            self.GetStress = sig
            # -----------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type C.1.2:
    '''  
    class PureCompressionLublinerExpoSoft(object):
        '''
           Tension:        Not considered in this case, since input strains and
                           stresses are only in pure compression state
           Compression:    Lubliner Yield Surface
                           Exponential Softening
        '''
        def GetStress(sig_eff, vars_le, vars_nl):
            e = vars_le['E']
            fcp = vars_nl['SP']
            fcbi = vars_nl['SBI']
            gc = vars_nl['GC']
            ft = vars_nl['FT']
            
            '''
               Compression Part
            '''            
            tau_neg = YieldCriterion.Lubliner.NegativeEquivalentStress\
                      (sig_eff, fcp, fcp, fcbi, ft)
            d_neg =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_neg, e, fcp, gc)
            sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
            sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)
            
            return sig_neg
# -----------------------------------------------------------------------------
    '''
       Type C.1.3:
    '''
    class PureTensionLublinerExpoSoft(object):
        '''
           Tension:        Lubliner Yield Surface
                           Exponential Softening
           Compression:    Not considered in this case, since input strains and
                           stresses are only in pure tension state
        '''
        def GetStress(sig_eff, vars_le, vars_nl):
            e = vars_le['E']
            fcp = vars_nl['SP']
            fcbi = vars_nl['SBI']
            ft = vars_nl['FT']
            gt = vars_nl['GT']
            
            '''
               Tension Part
            '''
            tau_pos = YieldCriterion.Lubliner.PositiveEquivalentStress\
                      (sig_eff, fcp, fcbi, ft)
            d_pos =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_pos, e, ft, gt)
            sig_eff_pos = EffectiveStressSplit.GetPositiveStress(sig_eff)
            sig_pos = ConstitutiveLaw._Compute_Stress(d_pos, sig_eff_pos)

            return sig_pos
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type C.2:
    '''
# -----------------------------------------------------------------------------
    '''
       Type C.2.1:
    '''
    class PosLublinerExpoSoftNegLublinerParHardExpoSoft(object):
        '''
           Tension:        Lubliner Yield Surface
                           Exponential Softening
           Compression:    Lubliner Yield Surface
                           Parabolic Hardening & Exponential Softening
        '''
        def __init__(self, sig_eff, vars_le, vars_nl):
            e    = vars_le['E']
            fcp  = vars_nl['SP']
            fcbi = vars_nl['SBI']
            gc   = vars_nl['GC']
            ft   = vars_nl['FT']
            gt   = vars_nl['GT']
            r0   = vars_nl['S0']
            re   = fcp
            rp   = vars_nl['SPP']

            '''
               Tension Part
            '''
            with tf.name_scope("PosEquiStress"):
                tau_pos = YieldCriterion.Lubliner.PositiveEquivalentStress\
                      (sig_eff, fcp, fcbi, ft)
            with tf.name_scope("PosDamVar"):
                d_pos =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_pos, e, ft, gt)
            with tf.name_scope("PosStressVec"):
                sig_eff_pos = EffectiveStressSplit.GetPositiveStress(sig_eff)
                sig_pos = ConstitutiveLaw._Compute_Stress(d_pos, sig_eff_pos)
            
            '''
               Compression Part
            '''            
            with tf.name_scope("NegEquiStress"):
                tau_neg = YieldCriterion.Lubliner.NegativeEquivalentStress\
                      (sig_eff, r0, fcp, fcbi, ft)
            with tf.name_scope("NegDamVar"):
                d_neg   = SofteningType.ParabolicHardeningExponentialSoftening.\
                      GetDamageVariable(tau_neg, e, fcp, gc, r0, re, rp)
            with tf.name_scope("PosStressVec"):
                sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
                sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)
            
            '''
               Total Part
            '''
            with tf.name_scope("TotalStressVec"):
                sig = tf.add(sig_pos, sig_neg)
            
            # -----------------------------------------------------------------
            # Caller ----------------------------------------------------------
            self.GetStress = sig
            # -----------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type C.2.2:
    '''  
    class PureCompressionLublinerParHardExpoSoft(object):
        '''
           Tension:        Not considered in this case, since input strains and
                           stresses are only in pure compression state
           Compression:    Lubliner Yield Surface
                           Parabolic Hardening & Exponential Softening
        '''
        def GetStress(sig_eff, vars_le, vars_nl):
            e    = vars_le['E']
            fcp  = vars_nl['SP']
            fcbi = vars_nl['SBI']
            gc   = vars_nl['GC']
            ft   = vars_nl['FT']
            gt   = vars_nl['GT']
            r0   = vars_nl['S0']
            re   = fcp
            rp   = vars_nl['SPP']
            
            '''
               Compression Part
            '''            
            tau_neg = YieldCriterion.Lubliner.NegativeEquivalentStress\
                      (sig_eff, r0, fcp, fcbi, ft)
            d_neg   = SofteningType.ParabolicHardeningExponentialSoftening.\
                      GetDamageVariable(tau_neg, e, fcp, gc, r0, re, rp)
            sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
            sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)
            
            return sig_neg
# -----------------------------------------------------------------------------
    '''
       Type C.2.3 = C.1.3
    '''
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type C.3:
    '''
# -----------------------------------------------------------------------------
    '''
       Type C.3.1:
    '''
    class PosLublinerExpoSoftNegLublinerBezierHardSoft(object):
        '''
           Tension:        Lubliner Yield Surface
                           Exponential Softening
           Compression:    Lubliner Yield Surface
                           Bezier Hardening & Softening
        '''
        def __init__(self, sig_eff, vars_le, vars_nl):
            e = vars_le['E']
            e0 = vars_nl['E0']
            ei = vars_nl['EI']
            ep = vars_nl['EP']
            ej = vars_nl['EJ']
            ek = vars_nl['EK']
            er = vars_nl['ER']
            eu = vars_nl['EU']
            s0 = vars_nl['S0']
            si = vars_nl['SI']
            sp = vars_nl['SP']
            sj = vars_nl['SJ']
            sk = vars_nl['SK']
            sr = vars_nl['SR']
            su = vars_nl['SU']
            fcbi = vars_nl['SBI']
            ft = vars_nl['FT']
            gt = vars_nl['GT']
            '''
               Tension Part
            '''
            with tf.name_scope("PosEquiStress"):
                tau_pos = YieldCriterion.Lubliner.PositiveEquivalentStress\
                      (sig_eff, sp, fcbi, ft)
            with tf.name_scope("PosDamVar"):
                d_pos =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_pos, e, ft, gt)
            with tf.name_scope("PosStressVec"):
                sig_eff_pos = EffectiveStressSplit.GetPositiveStress(sig_eff)
                sig_pos = ConstitutiveLaw._Compute_Stress(d_pos, sig_eff_pos)
            
            '''
               Compression Part
            '''            
            with tf.name_scope("NegEquiStress"):
                tau_neg = YieldCriterion.Lubliner.NegativeEquivalentStress\
                      (sig_eff, s0, sp, fcbi, ft)
            with tf.name_scope("NegDamVar"):
                d_neg   = SofteningType.BezierHardeningSoftening.GetDamageVariable\
                      (tau_neg, e, e0, ei, ep, ej, ek, er, eu, \
                       s0, si, sp, sj, sk, sr, su)
            with tf.name_scope("NegStressVec"):
                sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
                sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)
            
            '''
               Total Part
            '''
            with tf.name_scope("TotalStressVec"):
                sig = tf.add(sig_pos, sig_neg)
            
            # -----------------------------------------------------------------
            # Caller ----------------------------------------------------------
            self.GetStress = sig
            # -----------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type C.3.2:
    '''
    class PureCompressionLublinerBezierHardSoft(object):
        '''
           Tension:        Not considered in this case, since input strains and
                           stresses are only in pure compression state
           Compression:    Lubliner Yield Surface
                           Bezier Hardening & Softening
        '''
        def GetStress(sig_eff, vars_le, vars_nl):
            e = vars_le['E']
            e0 = vars_nl['E0']
            ei = vars_nl['EI']
            ep = vars_nl['EP']
            ej = vars_nl['EJ']
            ek = vars_nl['EK']
            er = vars_nl['ER']
            eu = vars_nl['EU']
            s0 = vars_nl['S0']
            si = vars_nl['SI']
            sp = vars_nl['SP']
            sj = vars_nl['SJ']
            sk = vars_nl['SK']
            sr = vars_nl['SR']
            su = vars_nl['SU']
            fcbi = vars_nl['SBI']
            ft = vars_nl['FT']
            gt = vars_nl['GT']
            '''
               Compression Part
            '''            
            tau_neg = YieldCriterion.Lubliner.NegativeEquivalentStress\
                      (sig_eff, s0, sp, fcbi, ft)
            d_neg   = SofteningType.BezierHardeningSoftening.GetDamageVariable\
                      (tau_neg, e, e0, ei, ep, ej, ek, er, eu, \
                       s0, si, sp, sj, sk, sr, su)
            sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
            sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)
            
            return sig_neg
# -----------------------------------------------------------------------------
    '''
       Type C.3.3  = Type C.1.3
    '''

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
        Type D:
        All Classes with Petracca Modified Yield
    '''
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type D.1:
    '''
# -----------------------------------------------------------------------------
    '''
       Type D.1.1:
    '''
    class PosPetraccaExpoSoftNegPetraccaExpoSoft(object):
        '''
           Tension:        Petracca Yield Surface
                           Exponential Softening
           Compression:    Petracca Yield Surface
                           Exponential Softening
        '''
        def __init__(self, sig_eff, vars_le, vars_nl):
            e = vars_le['E']
            fcp = vars_nl['SP']
            fcbi = vars_nl['SBI']
            gc = vars_nl['GC']
            ft = vars_nl['FT']
            gt = vars_nl['GT']
            
            '''
               Tension Part
            '''
            with tf.name_scope("PosEquiStress"):
                tau_pos = YieldCriterion.Petracca.PositiveEquivalentStress\
                      (sig_eff, fcp, fcbi, ft)
            with tf.name_scope("PosDamVar"):
                d_pos =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_pos, e, ft, gt)
            with tf.name_scope("PosStressVec"):
                sig_eff_pos = EffectiveStressSplit.GetPositiveStress(sig_eff)
                sig_pos = ConstitutiveLaw._Compute_Stress(d_pos, sig_eff_pos)
            
            '''
               Compression Part
            '''            
            with tf.name_scope("NegEquiStress"):
                tau_neg = YieldCriterion.Petracca.NegativeEquivalentStress\
                      (sig_eff, fcp, fcp, fcbi, ft)
            with tf.name_scope("NegDamVar"):
                d_neg =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_neg, e, fcp, gc)
            with tf.name_scope("NegStressVec"):
                sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
                sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)
            
            '''
               Total Part
            '''
            with tf.name_scope("TotalStressVec"):
                sig = tf.add(sig_pos, sig_neg)

            # -----------------------------------------------------------------
            # Caller ----------------------------------------------------------
            self.GetStress = sig
            # -----------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type D.1.2:
    '''            
    class PureCompressionPetraccaExpoSoft(object):
        '''
           Tension:        Not considered in this case, since input strains and
                           stresses are only in pure compression state
           Compression:    Petracca Yield Surface
                           Exponential Softening
        '''
        def GetStress(sig_eff, vars_le, vars_nl):
            e = vars_le['E']
            fcp = vars_nl['SP']
            fcbi = vars_nl['SBI']
            gc = vars_nl['GC']
            ft = vars_nl['FT']
            
            '''
               Compression Part
            '''            
            tau_neg = YieldCriterion.Petracca.NegativeEquivalentStress\
                      (sig_eff, fcp, fcp, fcbi, ft)
            d_neg =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_neg, e, fcp, gc)
            sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
            sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)
            
            return sig_neg
# -----------------------------------------------------------------------------
    '''
       Type D.1.3:
    '''
    class PureTensionPetraccaExpoSoft(object):
        '''
           Tension:        Lubliner Yield Surface
                           Exponential Softening
           Compression:    Not considered in this case, since input strains and
                           stresses are only in pure tension state
        '''
        def GetStress(sig_eff, vars_le, vars_nl):
            e = vars_le['E']
            fcp = vars_nl['SP']
            fcbi = vars_nl['SBI']
            ft = vars_nl['FT']
            gt = vars_nl['GT']
            
            '''
               Tension Part
            '''
            tau_pos = YieldCriterion.Petracca.PositiveEquivalentStress\
                      (sig_eff, fcp, fcbi, ft)
            d_pos =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_pos, e, ft, gt)
            sig_eff_pos = EffectiveStressSplit.GetPositiveStress(sig_eff)
            sig_pos = ConstitutiveLaw._Compute_Stress(d_pos, sig_eff_pos)

            return sig_pos
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type D.2:
    '''
# -----------------------------------------------------------------------------
    '''
       Type D.2.1:
    '''
    class PosPetraccaExpoSoftNegPetraccaParHardExpoSoft(object):
        '''
           Tension:        Petracca Yield Surface
                           Exponential Softening
           Compression:    Petracca Yield Surface
                           Parabolic Hardening & Exponential Softening
        '''
        def __init__(self, sig_eff, vars_le, vars_nl):
            e    = vars_le['E']
            fcp  = vars_nl['SP']
            fcbi = vars_nl['SBI']
            gc   = vars_nl['GC']
            ft   = vars_nl['FT']
            gt   = vars_nl['GT']
            r0   = vars_nl['S0']
            re   = fcp
            rp   = vars_nl['SPP']
            
            '''
               Tension Part
            '''
            with tf.name_scope("PosEquiStress"):
                tau_pos = YieldCriterion.Petracca.PositiveEquivalentStress\
                      (sig_eff, fcp, fcbi, ft)
            with tf.name_scope("PosDamVar"):
                d_pos =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_pos, e, ft, gt)
            with tf.name_scope("PosStressVec"):
                sig_eff_pos = EffectiveStressSplit.GetPositiveStress(sig_eff)
                sig_pos = ConstitutiveLaw._Compute_Stress(d_pos, sig_eff_pos)
            
            '''
               Compression Part
            '''            
            with tf.name_scope("NegEquiStress"):
                tau_neg = YieldCriterion.Petracca.NegativeEquivalentStress\
                      (sig_eff, r0, fcp, fcbi, ft)
            with tf.name_scope("NegDamVar"):
                d_neg   = SofteningType.ParabolicHardeningExponentialSoftening.\
                      GetDamageVariable(tau_neg, e, fcp, gc, r0, re, rp)
            with tf.name_scope("PosStressVec"):
                sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
                sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)
            
            '''
               Total Part
            '''
            with tf.name_scope("TotalStressVec"):
                sig = tf.add(sig_pos, sig_neg)
            # -----------------------------------------------------------------
            # Caller ----------------------------------------------------------
            self.GetStress = sig
            # -----------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type D.2.3 = D.1.3
    '''
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type D.3:
    '''
# -----------------------------------------------------------------------------
    '''
       Type D.3.1:
    '''
    class PosPetraccaExpoSoftNegPetraccaBezierHardSoft(object):
        '''
           Tension:        Petracca Yield Surface
                           Exponential Softening
           Compression:    Petracca Yield Surface
                           Bezier Hardening & Softening
        '''
        def __init__(self, sig_eff, vars_le, vars_nl):
            e = vars_le['E']
            e0 = vars_nl['E0']
            ei = vars_nl['EI']
            ep = vars_nl['EP']
            ej = vars_nl['EJ']
            ek = vars_nl['EK']
            er = vars_nl['ER']
            eu = vars_nl['EU']
            s0 = vars_nl['S0']
            si = vars_nl['SI']
            sp = vars_nl['SP']
            sj = vars_nl['SJ']
            sk = vars_nl['SK']
            sr = vars_nl['SR']
            su = vars_nl['SU']
            fcbi = vars_nl['SBI']
            ft = vars_nl['FT']
            gt = vars_nl['GT']
            '''
               Tension Part
            '''
            with tf.name_scope("PosEquiStress"):
               tau_pos = YieldCriterion.Petracca.PositiveEquivalentStress\
                      (sig_eff, sp, fcbi, ft)
            with tf.name_scope("PosDamVar"):
               d_pos =   SofteningType.ExponentialSoftening.GetDamageVariable\
                      (tau_pos, e, ft, gt)
            with tf.name_scope("PosStressVec"):
               sig_eff_pos = EffectiveStressSplit.GetPositiveStress(sig_eff)
               sig_pos = ConstitutiveLaw._Compute_Stress(d_pos, sig_eff_pos)
            
            '''
               Compression Part
            '''
            with tf.name_scope("NegEquiStress"):           
               tau_neg = YieldCriterion.Petracca.NegativeEquivalentStress\
                      (sig_eff, s0, sp, fcbi, ft)
            with tf.name_scope("NegDamVar"):
               d_neg   = SofteningType.BezierHardeningSoftening.GetDamageVariable\
                      (tau_neg, e, e0, ei, ep, ej, ek, er, eu, \
                       s0, si, sp, sj, sk, sr, su)
            with tf.name_scope("NegStressVector"):
               sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
               sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)

            '''
               Total Part
            '''
            with tf.name_scope("TotalStressVec"):
               sig = tf.add(sig_pos, sig_neg)
            
            # -----------------------------------------------------------------
            # Caller
            self.GetStress = sig
            # -----------------------------------------------------------------
# -----------------------------------------------------------------------------
    '''
       Type D.3.2:
    '''
    class PureCompressionPetraccaBezierHardSoft(object):
        '''
           Tension:        Not considered in this case, since input strains and
                           stresses are only in pure compression state
           Compression:    Petracca Yield Surface
                           Bezier Hardening & Softening
        '''
        def GetStress(sig_eff, vars_le, vars_nl):
            e = vars_le['E']
            e0 = vars_nl['E0']
            ei = vars_nl['EI']
            ep = vars_nl['EP']
            ej = vars_nl['EJ']
            ek = vars_nl['EK']
            er = vars_nl['ER']
            eu = vars_nl['EU']
            s0 = vars_nl['S0']
            si = vars_nl['SI']
            sp = vars_nl['SP']
            sj = vars_nl['SJ']
            sk = vars_nl['SK']
            sr = vars_nl['SR']
            su = vars_nl['SU']
            fcbi = vars_nl['SBI']
            ft = vars_nl['FT']
            gt = vars_nl['GT']
            '''
               Compression Part
            '''            
            tau_neg = YieldCriterion.Petracca.NegativeEquivalentStress\
                      (sig_eff, s0, sp, fcbi, ft)
            d_neg   = SofteningType.BezierHardeningSoftening.GetDamageVariable\
                      (tau_neg, e, e0, ei, ep, ej, ek, er, eu, \
                       s0, si, sp, sj, sk, sr, su)
            sig_eff_neg = EffectiveStressSplit.GetNegativeStress(sig_eff)
            sig_neg = ConstitutiveLaw._Compute_Stress(d_neg, sig_eff_neg)
            
            return sig_neg
# -----------------------------------------------------------------------------
    '''
       Type D.3.3  = Type D.1.3
    '''


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Utility Functions -----------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def _Compute_Stress(d, s_eff):
        with tf.name_scope("Apply1MinusD"):
            s = tf.multiply(tf.subtract(1.0, tf.expand_dims(d,1)), s_eff)
        return s 

    def _Elasticity_Tensor_Plane_Stress(nu,e):
        pos_11 = tf.divide(e,tf.subtract(1.0,tf.square(nu)))
        pos_12 = tf.divide(tf.multiply(e,nu),tf.subtract(1.0,tf.square(nu)))
        pos_33 = tf.divide(e,tf.multiply(2.0,tf.add(1.0,nu)))
        c_temp = [[pos_11, pos_12, 0.0     ], \
                  [pos_12, pos_11, 0.0     ],\
                  [0.0     , 0.0     , pos_33]]
        c = tf.stack(c_temp)
        return c

    def _Elasticity_Tensor_Plane_Strain(nu,e):
        pos_11 = tf.divide( \
            tf.multiply(tf.subtract(1.0,nu),e), \
            tf.multiply(tf.add(1.0,nu), \
                        tf.subtract(1.0,tf.multiply(2.0,nu) ) ))
        pos_12 = tf.divide( \
            tf.multiply(nu,e), \
            tf.multiply(tf.add(1.0,nu), \
                        tf.subtract(1.0,tf.multiply(2.0,nu) ) ))
        pos_33 = tf.divide(e,tf.multiply(2.0,tf.add(1.0,nu)))
        c_temp = [[pos_11, pos_12, 0.0     ], \
                  [pos_12, pos_11, 0.0     ],\
                  [0.0     , 0.0     , pos_33]]
        c = tf.stack(c_temp)
        return c
        
