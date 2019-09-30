import tensorflow as tf

class ModelVariables(object):
# ----------------------------------------------------------------------------
    '''
       Call the Linear Elastic Property Variables
          A) Define Variables
          B) Constrain Variables
    '''
    class LinearElastic(object):
        def __init__(self, initial_values):
            entries = initial_values['LinearElastic']
            with tf.name_scope('ModelVariables'):
                e = tf.Variable(entries['YoungsModulusTemp']['initial_value'], \
                name='YoungsModulusTemp')
                e_multiplier = tf.constant(entries['YoungsModulusMultiplier'], \
                name='YoungsModulusMultiplier')
                nu = tf.Variable(entries['PoissonsRatio']['initial_value'],name='PoissonsRatio')
            self.Variables = [e, nu]
            self.Vars4Print = ['E:   Youngs Modulus', 'Nu:  Poissons Ratio']
            self.LearningRates = [
                entries['YoungsModulusTemp']['learning_rate'], \
                entries['PoissonsRatio']['initial_value']
            ]
    
        def ConstrainVariables(variables, initial_values):
            entries = initial_values['LinearElastic']
            with tf.name_scope('ConstrainLinearElasticVariables'):
                e = tf.multiply(variables[0], \
                                entries['YoungsModulusMultiplier'],\
                                name='YoungsModulus')
                nu  = tf.clip_by_value(variables[1], 0.0000001, 0.5)
            var_dict = {'E':e, 'NU': nu}
            return var_dict
    '''
       Additional Class to initialize the Tensorflow Summaries of the Linear 
       Elastic Properties 
    '''
    class LinearElasticSummary(object):
        def __init__(self, variables):
            with tf.name_scope('LinearElastic'):
                tf.summary.scalar('YoungsModulus', variables['E'])
                tf.summary.scalar('PoissonsRatio', variables['NU'])
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
    '''
       Call the Property Variables for the TenLinSoft and CompLinSoft
          A) Define Variables
          B) Constrain Variables
    '''
    class LinSoftLinSoft(object):
        def __init__(self, initial_values):
            entries = initial_values['TensLinSoftCompLinSoft']
            with tf.name_scope('DamageModelVariables'):
                fcp = tf.Variable(entries['CompressiveStrength']['initial_value'], \
                      name='CompressiveStrength')
                fcpb = tf.Variable(entries['CompressiveBoundingStress']['initial_value'], \
                      name='CompressiveBoundingStress')
                fcbi = tf.Variable(entries['BiaxialCompressiveStrength']['initial_value'], \
                      name='BiaxialCompressiveStrength')     
                ft  = tf.Variable(entries['TensionStrength']['initial_value'], \
                      name='TensionStrength')
                ftb = tf.Variable(entries['TensionBoundingStress']['initial_value'], \
                      name='TensionBoundingStress') 
            self.Variables = [fcp, fcpb, ft, ftb, fcbi]
            self.Vars4Print = ['Fcp:  Compressive Strength', \
                               'Fcpb: Compressive Bounding Stress', \
                               'Fcbi: Biaxial Compressive Strength',\
                               'Ft:   Tensile Strength',\
                               'Ftb:  Tensile Bounding Stress']
            self.LearningRates = [                                     \
                entries['CompressiveStrength']['learning_rate'],       \
                entries['CompressiveBoundingStress']['learning_rate'], \
                entries['TensionStrength']['learning_rate'],           \
                entries['TensionBoundingStress']['learning_rate'],     \
                entries['BiaxialCompressiveStrength']['learning_rate'] \
            ]

        def ConstrainVariables(variables, initial_values):
            entries = initial_values['TensLinSoftCompLinSoft']
            with tf.name_scope('ConstrainDamageVariables'):
                tol_s = 1e-10
                tol_g = 1e-10
                inf = 1e+12

                fcp  = tf.clip_by_value(variables[0],tol_s, inf)
                fcpb = tf.clip_by_value(variables[1],tf.add(fcp, tol_s), inf)
                fcbi = tf.clip_by_value(variables[4],tf.add(fcp, tol_s), inf)
                ft   = tf.clip_by_value(variables[2],tol_s, inf)
                ftb  = tf.clip_by_value(variables[3],tf.add(ft, tol_s), inf)
            var_dict = {'FCP': fcp, 'FCPB': fcpb, 'FT': ft, 'FTB': ftb, 'FCBI': fcbi}
            return var_dict

    '''
       Additional Class to initialize the Tensorflow Summaries of the 
       TenExpoSoft and CompExpoSoft Properties 
    '''        
    class LinSoftLinSoftSummary(object):
        def __init__(self, variables):
            with tf.name_scope('DamageParameters'):
                tf.summary.scalar('CompressiveStrength', variables['FCP'])
                tf.summary.scalar('CompressiveBoundingStress', variables['FCPB'])
                tf.summary.scalar('TensionStrength', variables['FT'])
                tf.summary.scalar('TensionBoundingStress', variables['FTB'])
                tf.summary.scalar('BiaxialCompressiveStrength', variables['FCBI'])

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
    '''
       Call the Property Variables for the TenExpoSoft and CompExpoSoft
          A) Define Variables
          B) Constrain Variables
    '''
    class ExpoSoftExpoSoft(object):
        def __init__(self, initial_values):
            entries = initial_values['TensExpoSoftCompExpoSoft']
            with tf.name_scope('DamageModelVariables'):
                sp  = tf.Variable(entries['CompressiveStrength']['initial_value'], \
                      name='CompressiveStrength')
                sbi = tf.Variable(entries['BiaxialCompressiveStrength']['initial_value'], \
                      name='BiaxialCompressiveStrength')
                gc  = tf.Variable(entries['CompressiveFractureEnergy']['initial_value'], \
                      name='CompressiveFractureEnergy')
                ft  = tf.Variable(entries['TensionStrength']['initial_value'], \
                      name='TensionStrength')
                gt  = tf.Variable(entries['TensionFractureEnergy']['initial_value'], \
                      name='TensionFractureEnergy')
            self.Variables = [sp, sbi, gc, ft, gt]
            self.Vars4Print = ['Sp:  Compressive Strength', \
                               'Sbi: Biaxial Compressive Strength',\
                               'Gc:  Fracture Energy Compression', \
                               'Ft:  Tensile Strength',\
                               'Gt:  Fracture Energy Tension']
            self.LearningRates = [                                      \
                entries['CompressiveStrength']['learning_rate'],        \
                entries['BiaxialCompressiveStrength']['learning_rate'], \
                entries['CompressiveFractureEnergy']['learning_rate'],  \
                entries['TensionStrength']['learning_rate'],            \
                entries['TensionFractureEnergy']['learning_rate']       \
            ]

        def ConstrainVariables(variables, initial_values):
            entries = initial_values['TensExpoSoftCompExpoSoft']
            with tf.name_scope('ConstrainDamageVariables'):
                tol_s = 1e-10
                tol_g = 1e-10
                inf = 1e+12
                energy_multi = entries['EnergyMultiplier']
                stress_multi = entries['StressMultiplier']

                sp  = tf.clip_by_value(tf.multiply(variables[0],stress_multi),\
                                                   tol_s, inf)
                sbi = tf.clip_by_value(tf.multiply(variables[1],stress_multi),\
                                                   tf.add(sp, tol_s), inf)
                gc  = tf.clip_by_value(tf.multiply(variables[2],energy_multi),\
                                                   tol_g, inf)
                ft  = tf.clip_by_value(tf.multiply(variables[3],stress_multi),\
                                                   tol_s, inf)
                gt  = tf.clip_by_value(tf.multiply(variables[4],energy_multi),\
                                                   tol_g, inf)
                
            var_dict = {'SP': sp, 'SBI': sbi, 'GC': gc, 'FT': ft, 'GT': gt}
            return var_dict
    '''
       Additional Class to initialize the Tensorflow Summaries of the 
       TenExpoSoft and CompExpoSoft Properties 
    '''        
    class ExpoSoftExpoSoftSummary(object):
        def __init__(self, variables):
            with tf.name_scope('DamageParameters'):
                tf.summary.scalar('CompressiveStrength', variables['SP'])
                tf.summary.scalar('BiaxialCompressiveStrength', variables['SBI'])
                tf.summary.scalar('CompressiveFractureEnergy', variables['GC'])
                tf.summary.scalar('TensionStrength', variables['FT'])
                tf.summary.scalar('TensionFractureEnergy', variables['GT'])

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
    '''
       Call the Property Variables for the TenExpoSoft and CompParHardExpoSoft
          A) Define Variables
          B) Constrain Variables
    '''
    class ExpoSoftParHardExpoSoft(object):
        def __init__(self, initial_values):
            entries = initial_values['TensExpoSoftCompParHardExpoSoft']
            with tf.name_scope('DamageModelVariables'):
                s0  = tf.Variable(entries['CompressiveElasticLimit']['initial_value'], \
                      name='ElasticCompressionLimit')
                sp  = tf.Variable(entries['CompressiveStrength']['initial_value'],\
                      name='CompressiveStrength')
                spp = tf.Variable(entries['CompressiveVirtualPeakStrength']['initial_value'],\
                      name='CompressiveVirtualPeakStrength')
                sbi = tf.Variable(entries['BiaxialCompressiveStrength']['initial_value'],\
                      name='BiaxialCompressiveStrength')
                gc  = tf.Variable(entries['CompressiveFractureEnergy']['initial_value'],\
                      name='CompressiveFractureEnergy')
                ft  = tf.Variable(entries['TensionStrength']['initial_value'],\
                      name='TensionStrength')
                gt  = tf.Variable(entries['TensionFractureEnergy']['initial_value'],\
                      name='TensionFractureEnergy')
            self.Variables = [s0, sp, spp, sbi, gc, ft, gt]
            self.Vars4Print = ['S0:  Compressive Elastic Limit',\
                               'Sp:  Compressive Strength',
                               'Spp: Compressive Virtual Peak Strength',\
                               'Sbi: Biaxial Compressive Strength',\
                               'Gc:  Fracture Energy Compression', \
                               'Ft:  Tensile Strength',\
                               'Gt:  Fracture Energy Tension']
            self.LearningRates = [                                             \
                entries['CompressiveElasticLimit']['learning_rate'],           \
                entries['CompressiveStrength']['learning_rate'],               \
                entries['Compressive Virtual Peak Strength']['learning_rate'], \
                entries['BiaxialCompressiveStrength']['learning_rate'],        \
                entries['CompressiveFractureEnergy']['learning_rate'],         \
                entries['TensionStrength']['learning_rate'],                   \
                entries['TensionFractureEnergy']['learning_rate']              \
            ]

        def ConstrainVariables(variables,initial_values):
            entries = initial_values['TensExpoSoftCompParHardExpoSoft']
            with tf.name_scope('ConstrainDamageVariables'):
                tol_s = 1e-10
                tol_g = 1e-10
                inf = 1e+12

                energy_multi = entries['EnergyMultiplier']
                stress_multi = entries['StressMultiplier']

                s0  = tf.clip_by_value(tf.multiply(variables[0],stress_multi),\
                                                   tol_s, inf)
                sp  = tf.clip_by_value(tf.multiply(variables[1],stress_multi),\
                                                   tf.add(s0,tol_s), inf)
                spp = tf.clip_by_value(tf.multiply(variables[2],stress_multi),\
                                                   tf.add(sp,tol_s), inf)
                sbi = tf.clip_by_value(tf.multiply(variables[3],stress_multi),\
                                                   tf.add(sp, tol_s), inf)
                gc  = tf.clip_by_value(tf.multiply(variables[4],energy_multi),\
                                                   tol_g, inf)
                ft  = tf.clip_by_value(tf.multiply(variables[5],stress_multi),\
                                                   tol_s, inf)
                gt  = tf.clip_by_value(tf.multiply(variables[6],energy_multi),\
                                                   tol_g, inf)
            var_dict = {'S0': s0, 'SP': sp, 'SPP': spp, 'SBI': sbi, 'GC': gc, \
                        'FT': ft, 'GT': gt}
            return var_dict

    '''
       Additional Class to initialize the Tensorflow Summaries of the 
       TenExpoSoft and CompParHardExpoSoft Properties 
    '''      
    class ExpoSoftParHardExpoSoftSummary(object):
        def __init__(self, variables):
            with tf.name_scope('DamageParameters'):
                tf.summary.scalar('CompressiveElasticLimit', \
                                  variables['S0'])
                tf.summary.scalar('CompressiveStrength', \
                                  variables['SP'])
                tf.summary.scalar('CompressiveVirtualPeakStrength', \
                                  variables['SPP'])
                tf.summary.scalar('BiaxialCompressiveStrength', \
                                  variables['SBI'])
                tf.summary.scalar('CompressiveFractureEnergy', \
                                  variables['GC'])
                tf.summary.scalar('TensionStrength', \
                                  variables['FT'])
                tf.summary.scalar('TensionFractureEnergy', \
                                  variables['GT'])

# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
    '''
       Call the Property Variables for the TenExpoSoft and CompBezierHardSoft 
       without Fracture Energy
          A) Define Variables
          B) Constrain Variables
    '''
    class ExpoSoftBezierHardSoft(object):
        def __init__(self, initial_values):
            entries = initial_values['TensExpoSoftCompBezierHardSoft']
            with tf.name_scope('DamageModelVariables'):
                ep = tf.Variable(entries['StrainCompressiveStrength']['initial_value'], \
                     name = 'StrainCompressiveStrength')
                ej = tf.Variable(entries['JcontrolCompressiveStrain']['initial_value'], \
                     name = 'JcontrolCompressiveStrain')
                ek = tf.Variable(entries['KcontrolCompressiveStrain']['initial_value'], \
                     name = 'KcontrolCompressiveStrain')
                eu = tf.Variable(entries['CompressiveUltimateStrain']['initial_value'], \
                     name = 'CompressiveUltimateStrain')
                s0 = tf.Variable(entries['CompressiveElasticLimit']['initial_value'], \
                     name='CompressiveElasticLimit')
                sp = tf.Variable(entries['CompressiveStrength']['initial_value'], \
                     name='CompressiveStrength')
                sk = tf.Variable(entries['KcontrolCompressiveStress']['initial_value'], \
                     name='KcontrolCompressiveStress')
                sr = tf.Variable(entries['CompressiveResidualStrength']['initial_value'], \
                     name = 'CompressiveResidualStrength')
                sbi = tf.Variable(entries['BiaxialCompressiveStrength']['initial_value'], \
                     name = 'BiaxialCompressiveStrength')
                ft = tf.Variable(entries['TensionStrength']['initial_value'],\
                     name='TensionStrength')
                gt = tf.Variable(entries['TensionFractureEnergy']['initial_value'],\
                     name='TensionFractureEnergy')

            self.Variables = [ep, ej, ek, eu, s0, sp, sk, sr, sbi, ft, gt]
            self.Vars4Print = ['Ep:  Strain Compressive Strength', \
                               'Ej:  J-Control Compressive Strain',\
                               'Ek:  K-Control Compressive Strain', \
                               'Eu:  Compressive Ultimate Strain',\
                               'S0:  Compressive Elastic Limit', \
                               'Sp:  Compressive Strength', \
                               'Sk:  K-Control Compressive Strength', \
                               'Sr:  Compressive Residual Strength',\
                               'Sbi: Biaxial Compressive Strength', \
                               'Ft:  Tensile Strength',\
                               'Gt:  Fracture Energy Tension']
            self.LearningRates = [                                       \
                entries['StrainCompressiveStrength']['learning_rate'],   \
                entries['JcontrolCompressiveStrain']['learning_rate'],   \
                entries['KcontrolCompressiveStrain']['learning_rate'],   \
                entries['CompressiveUltimateStrain']['learning_rate'],   \
                entries['CompressiveElasticLimit']['learning_rate'],     \
                entries['CompressiveStrength']['learning_rate'],         \
                entries['KcontrolCompressiveStress']['learning_rate'],   \
                entries['CompressiveResidualStrength']['learning_rate'], \
                entries['BiaxialCompressiveStrength']['learning_rate'],  \
                entries['TensionStrength']['learning_rate'],             \
                entries['TensionFractureEnergy']['learning_rate']        \
            ]

        def ConstrainVariables(variables, variables_le, initial_values):
            with tf.name_scope('ConstrainDamageVariables'):
                entries = initial_values['TensExpoSoftCompBezierHardSoft']

                tol_s = 1e-10
                tol_e = 1e-10
                tol_eps = 1e-10
                tol_g = 1e-10
                inf   = 1e+12

                stress_multi = entries["StressMultiplier"]
                energy_multi = entries["EnergyMultiplier"]

                sp  = tf.clip_by_value(tf.multiply(variables[5],stress_multi), tol_s, inf)
                #sbi = tf.clip_by_value(tf.multiply(variables[8],stress_multi), tol_s, inf)
                sbi = tf.clip_by_value(tf.multiply(variables[8],stress_multi), sp * 1.05, inf)
                s0  = tf.clip_by_value(tf.multiply(variables[4],stress_multi), tol_s, tf.subtract(sp, tol_s))

                si = sp
                sj = sp
                
                sr = tf.clip_by_value(tf.multiply(variables[7],stress_multi), tol_s, tf.multiply(sp,0.9))
                sk = tf.clip_by_value(tf.multiply(variables[6],stress_multi), tf.add(sr, tol_s), tf.subtract(sj, tol_s))
                su = sr

                e0 = tf.divide(s0, variables_le['E'])
                ei = tf.divide(si, variables_le['E'])
                
                ep = tf.clip_by_value(variables[0], tf.add(ei,tol_e), inf)
                ej = tf.clip_by_value(variables[1], tf.add(ep,tol_e), inf)
                ek = tf.clip_by_value(variables[2], tf.add(ej,tol_e), inf)

                #er_t1 = tf.multiply(tf.subtract(ek, ej),tf.subtract(sp,sr))
                #er_t2 = tf.divide(er_t1,tf.subtract(sp,sk))
                #er_t3 = tf.add(ej,er_t2)
                #er = tf.clip_by_value(er_t3, tf.add(ek,tol_e), inf)

                er_t1 = tf.multiply(tf.subtract(ek, ej),tf.subtract(sp,sr), name ="Er_temp1")
                er_t2 = tf.divide(er_t1,tf.subtract(sp,sk), name ="Er_temp2")
                er_t3 = tf.add(ej,er_t2, name ="RBezControlStrain")
                er = tf.clip_by_value(er_t3, tf.add(ek,tol_e), inf, name = "ClipEr")
                
                eu = tf.clip_by_value(variables[3], tf.add(er,tol_e), inf)
                    
                ft = tf.clip_by_value(tf.multiply(variables[9],stress_multi), tol_s, inf)
                gt = tf.clip_by_value(tf.multiply(variables[10], energy_multi), tol_g, inf)

                ei = ModelVariables._petracca_bezier_update(e0, ei, ep, "Ei_")
                ej = ModelVariables._petracca_bezier_update(ep, ej, ek, "Ej_")
                er = ModelVariables._petracca_bezier_update(ek, er, eu, "Er_")
                
            var_dict = {\
                'S0': s0, 'SI': si, 'SP': sp, 'SJ': sj, 'SR': sr, 'SK': sk, \
                'SU': su, 'SBI': sbi,\
                'E0': e0, 'EI': ei, 'EP': ep, 'EJ': ej, 'EK': ek, 'ER': er, \
                'EU': eu, \
                'FT': ft, 'GT': gt\
                }
            return var_dict
    '''
       Additional Class to initialize the Tensorflow Summaries of the 
       TenExpoSoft and CompBezHardSoft Properties 
    '''                      
    class ExpoSoftBezierHardSoftSummary(object):
        def __init__(self, variables):
            with tf.name_scope('DamageParameters'):
                tf.summary.scalar('StrainCompressiveStrength', \
                                  variables['EP'])
                tf.summary.scalar('JcontrolCompressiveStrain', \
                                  variables['EJ'])
                tf.summary.scalar('KcontrolCompressiveStrain', \
                                  variables['EK'])
                tf.summary.scalar('CompressiveUltimateStrain', \
                                  variables['EU'])
                tf.summary.scalar('CompressiveElasticLimit', \
                                  variables['S0'])
                tf.summary.scalar('CompressiveStrength', \
                                  variables['SP'])
                tf.summary.scalar('KcontrolCompressiveStress', \
                                  variables['SK'])
                tf.summary.scalar('CompressiveResidualStrength', \
                                  variables['SR'])
                tf.summary.scalar('BiaxialCompressiveStrength', \
                                  variables['SBI'])
                tf.summary.scalar('TensionStrength', \
                                  variables['FT'])
                tf.summary.scalar('TensionFractureEnergy', \
                                  variables['GT'])



# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
    '''
       Call the Property Variables for the TenExpoSoft and CompBezierHardSoft 
       with Controllers as Constant and with Fracture Energy
          A) Define Variables
          B) Constrain Variables
    '''               
    class ExpoSoftBezierHardSoftWithFractureEnergy(object):
        def __init__(self, initial_values):
            entries = initial_values['TensExpoSoftCompBezierHardSoftWithFractureEnergy']
            with tf.name_scope('DamageModelVariables'):
                ep = tf.Variable(entries['StrainCompressiveStrength']['initial_value'], \
                     name = 'StrainCompressiveStrength')
                s0 = tf.Variable(entries['CompressiveElasticLimit']['initial_value'], \
                     name='CompressiveElasticLimit')
                sp = tf.Variable(entries['CompressiveStrength']['initial_value'], \
                     name='CompressiveStrength')
                sr = tf.Variable(entries['CompressiveResidualStrength']['initial_value'], \
                     name = 'CompressiveResidualStrength')
                sbi = tf.Variable(entries['BiaxialCompressiveStrength']['initial_value'], \
                     name = 'BiaxialCompressiveStrength')
                gc = tf.Variable(entries['CompressiveFractureEnergy']['initial_value'], \
                     name = "CompressionFractureEnergy")
                ft = tf.Variable(entries['TensionStrength']['initial_value'], \
                     name='TensionStrength')
                gt = tf.Variable(entries['TensionFractureEnergy']['initial_value'], \
                     name='TensionFractureEnergy')
                c1 = tf.Variable(entries['BezierControllerC1']['initial_value'], name = 'C_1')
                c2 = tf.Variable(entries['BezierControllerC2']['initial_value'], name = 'C_2')
                c3 = tf.Variable(entries['BezierControllerC3']['initial_value'], name = 'C_3')

            self.Variables = [ep, s0, sp, sr, sbi, gc, ft, gt, c1, c2, c3]
            self.Vars4Print = ['Ep:  Strain Compressive Strength', 
                               'S0:  Compressive Elastic Limit', \
                               'Sp:  Compressive Strength', \
                               'Sr:  Compressive Residual Strength',\
                               'Sbi: Biaxial Compressive Strength', \
                               'Gc:  Fracture Energy Compression', \
                               'Ft:  Tensile Strength', \
                               'Gt:  Fracture Energy Tension', \
                               'C1:  BezierControllerC1', \
                               'C2:  BezierControllerC2', \
                               'C3:  BezierControllerC3']
            self.LearningRates = [                                       \
                entries['StrainCompressiveStrength']['learning_rate'],   \
                entries['CompressiveElasticLimit']['learning_rate'],     \
                entries['CompressiveStrength']['learning_rate'],         \
                entries['CompressiveResidualStrength']['learning_rate'], \
                entries['BiaxialCompressiveStrength']['learning_rate'],  \
                entries['CompressiveFractureEnergy']['learning_rate'],   \
                entries['TensionStrength']['learning_rate'],             \
                entries['TensionFractureEnergy']['learning_rate'],       \
                entries['BezierControllerC1']['learning_rate'],          \
                entries['BezierControllerC2']['learning_rate'],          \
                entries['BezierControllerC3']['learning_rate']           \
            ]

        def ConstrainVariables(variables, variables_le, initial_values):
            with tf.name_scope('ConstrainDamageVariables'):
                entries = initial_values['TensExpoSoftCompBezierHardSoftWithFractureEnergy']
                
                tol_s = tf.constant(1e-3, name="StressToleranceMinimum")
                tol_e = tf.constant(1e-4, name="StrainToleranceMinimum")
                tol_g = tf.constant(1e-8, name="EnergyToleranceMinimum")
                inf   = tf.constant(1e+12, name="ToleranceInfinity")

                stress_multi = tf.constant(entries["StressMultiplier"])
                energy_multi = tf.constant(entries["EnergyMultiplier"])
                cont_multi   = tf.constant(entries["ControllerMultiplier"])

                with tf.name_scope("BezierControllers"):
                    c1 = tf.clip_by_value(tf.multiply(variables[8],cont_multi),\
                         tol_e, tf.subtract(1.0,tol_e), name= "ClipC1")
                    c2 = tf.clip_by_value(tf.multiply(variables[9],cont_multi),\
                         0.51, 0.999, name="ClipC2")
                    c3 = tf.clip_by_value(tf.multiply(variables[10],cont_multi),\
                         0.5, 100.0, name="ClipC3")
                with tf.name_scope("CompressiveStrength"):
                    sp  = tf.clip_by_value(tf.multiply(variables[2],stress_multi),\
                          tol_s, inf, name = "ClipSp")
                    si = sp
                    sj = sp
                with tf.name_scope("BiaxialCompressiveStrength"):
                    sbi = tf.clip_by_value(tf.multiply(variables[4],stress_multi),\
                          tol_s, inf, name = "ClipSbi")
                with tf.name_scope("DamageOnsetStrengthCompression"):
                    s0  = tf.clip_by_value(tf.multiply(variables[1],stress_multi),\
                          tol_s, tf.subtract(sp, tol_s), name ="ClipS0")
                with tf.name_scope("ResidualCompressiveStrength"):
                    sr = tf.clip_by_value(tf.multiply(variables[3],stress_multi),\
                         tol_s, tf.multiply(sp,0.9), name = "ClipSr")
                    su = sr
                with tf.name_scope("KBezierControllerStress"):
                    sk = tf.add(sr, tf.multiply(tf.subtract(sp,sr), c1),\
                         name="KBezierControlStress")
                with tf.name_scope("StrainDamageOnsetCompressive"):
                    e0 = tf.divide(s0, variables_le['E'], name="DamageOnsetStrainCompression")
                with tf.name_scope("I_BezierControllStrain"):
                    ei = tf.divide(si, variables_le['E'], name="IBezControlStrain")
                with tf.name_scope("StrainCompressiveStrength"):
                    ep = tf.clip_by_value(variables[0], tf.add(ei,tol_e), inf, name="ClipEp")
                with tf.name_scope("J_K_BezierControllStrain"):
                    alpha = tf.multiply(2.0,tf.subtract(ep,tf.divide(sp, variables_le['E'])), name="AlphaBezController")
                    ej = tf.add(ep,tf.multiply(alpha,c2), name="JBezControlStrain")
                    ek = tf.add(ep, alpha, name = "KBezControlStrain")
                    #ek = tf.add(ej, tf.multiply(alpha,tf.subtract(1.0,c2)), name ="KBezControlStrain")
                with tf.name_scope("R_BezierControlStrain"):
                    er_t1 = tf.multiply(tf.subtract(ek, ej),tf.subtract(sp,sr), name ="Er_temp1")
                    er_t2 = tf.divide(er_t1,tf.subtract(sp,sk), name ="Er_temp2")
                    er_t3 = tf.add(ej,er_t2, name ="RBezControlStrain")
                    er = tf.clip_by_value(er_t3, tf.add(ek,tol_e), inf, name = "ClipEr")
                with tf.name_scope("U_BezierControlStrain"):
                    eu = tf.multiply(er,c3, name="UBezControlStrain")
                with tf.name_scope("FractureEnergyCompression"):
                    gc = tf.clip_by_value(tf.multiply(variables[5], energy_multi), tol_g, inf, name="ClipGc")

                with tf.name_scope("UpdateBezierControlStrains"):
                    with tf.name_scope("EnergStretchBezier"):
                        (ej, ek, er, eu) = \
                        ModelVariables._PetraccaBezierEnergyUpdate\
                        (gc, sp, sj, sk, sr, su, ep, ej, ek, er, eu)
                    with tf.name_scope("A-FactorBezierDiscussion"):
                        ei = ModelVariables._petracca_bezier_update(e0, ei, ep, "Ei_")
                        ej = ModelVariables._petracca_bezier_update(ep, ej, ek, "Ej_")
                        er = ModelVariables._petracca_bezier_update(ek, er, eu, "Er_")
                with tf.name_scope("TensionStrength"):
                    ft = tf.clip_by_value(tf.multiply(variables[6],\
                         stress_multi), tol_s, inf, name="ClipFt")
                with tf.name_scope("FractureEnergyTension"):
                    gt = tf.clip_by_value(tf.multiply(variables[7],\
                         energy_multi), tol_g, inf, name="ClipGt")
                
            var_dict = {\
                'S0': s0, 'SI': si, 'SP': sp, 'SJ': sj, \
                'SR': sr, 'SK': sk, 'SU': su, 'SBI': sbi,\
                'E0': e0, 'EI': ei, 'EP': ep, 'EJ': ej, \
                'EK': ek, 'ER': er, 'EU': eu, \
                'FT': ft, 'GT': gt, 'GC': gc,\
                'C1': c1, 'C2': c2, 'C3': c3\
                }
            return var_dict
                
    class ExpoSoftBezierHardSoftWithFractureEnergySummary(object):
        def __init__(self, variables):
            with tf.name_scope('DamageParameters'):
                tf.summary.scalar('StrainCompressiveStrength', variables['EP'])
                tf.summary.scalar('JcontrolCompressiveStrain', variables['EJ'])
                tf.summary.scalar('KcontrolCompressiveStrain', variables['EK'])
                tf.summary.scalar('CompressiveUltimateStrain', variables['EU'])

                tf.summary.scalar('CompressiveElasticLimit', variables['S0'])
                tf.summary.scalar('CompressiveStrength', variables['SP'])
                tf.summary.scalar('KcontrolCompressiveStress', variables['SK'])
                tf.summary.scalar('CompressiveResidualStrength', variables['SR'])
                tf.summary.scalar('BiaxialCompressiveStrength', variables['SBI'])

                tf.summary.scalar('TensionStrength', variables['FT'])
                tf.summary.scalar('TensionFractureEnergy', variables['GT'])

                tf.summary.scalar('BezierControllerC1', variables['C1'])
                tf.summary.scalar('BezierControllerC2', variables['C2'])
                tf.summary.scalar('BezierControllerC3', variables['C3'])
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Utility Functions -----------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
    def _petracca_bezier_update(x1,x2,x3, name):
        A = tf.subtract(tf.add(x1, x3),tf.multiply(2.0, x2), name= name + "AToCheck")
        alpha__ = tf.constant(1e-4)
        condition = tf.less(tf.abs(A),1.0e-12)
        x2_new = tf.add(x2,tf.multiply(tf.subtract(x3,x1),alpha__))
        x2_upd = tf.where(condition, x2_new, x2, name=name +"Updated")
        return (x2_upd)

    def _PetraccaBezierEnergyUpdate(gc, sp, sj, sk, sr, su, ep, ej, ek, er, eu):
        with tf.name_scope("ComputeBezEnerg"):
            gc1 = tf.divide(tf.multiply(sp,ep),2.0, name="Energy1")
            gc2 = ModelVariables.__bezier_energy(ep,ej,ek,sp,sj,sk)
            gc3 = ModelVariables.__bezier_energy(ek,er,eu,sk,sr,su)
            gc_bez = tf.add(tf.add(gc1,gc2),gc3, name = "EnergyTotal")
        with tf.name_scope("BezierStretcher"):
            stretch_temp1 = tf.subtract(gc,gc1)
            stretch_temp2 = tf.subtract(gc_bez, gc1)
            stretch_temp3 = tf.divide(stretch_temp1, stretch_temp2)
            stretcher = tf.subtract(stretch_temp3,1.0, name="Stretcher")
        with tf.name_scope("ApplyStretcher"):
            ej = ModelVariables.__apply_bezier_stretcher(stretcher, ej, ep) 
            ek = ModelVariables.__apply_bezier_stretcher(stretcher, ek, ep)
            er = ModelVariables.__apply_bezier_stretcher(stretcher, er, ep)
            eu = ModelVariables.__apply_bezier_stretcher(stretcher, eu, ep)
        return (ej, ek, er, eu)


    def __bezier_energy(x1, x2, x3, y1, y2, y3):
        gi_temp1 = tf.add(tf.divide(tf.multiply(x2,y1),3.0), tf.divide(tf.multiply(x3,y1),6.0))
        gi_temp2 = tf.subtract(tf.divide(tf.multiply(x3,y2),3.0), tf.divide(tf.multiply(x2,y3),3.0))
        gi_temp3 = tf.divide(tf.multiply(x3,y3),2.0)
        gi_temp4 = tf.multiply(x1, tf.add(tf.add(tf.divide(y1,2.0), tf.divide(y2,3.0)), tf.divide(y3,6.0)))
        gi = tf.add(tf.add(gi_temp1,gi_temp2), tf.subtract(gi_temp3, gi_temp4),  name="Energy2")
        return gi

    def __apply_bezier_stretcher(s, ei, ep):
        with tf.name_scope("StrainUpdate"):
            ei = tf.add(ei,tf.multiply(s,tf.subtract(ei,ep)))
        return ei


# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------
    '''
       Call the Property Variables for the TenExpoSoft and CompBezierHardSoft 
       with Controllers as Variables and without Fracture Energy
          A) Define Variables
          B) Constrain Variables
    '''
    class ExpoSoftBezierHardSoftControlled(object):
        def __init__(self, initial_values):
            entries = initial_values['TensExpoSoftCompBezierHardSoftControlled']
            with tf.name_scope('DamageModelVariables'):
                c1 = tf.Variable(entries['ControllerC1']['initial_value'], name = 'C_1')
                c2 = tf.Variable(entries['ControllerC2']['initial_value'], name = 'C_2')
                c3 = tf.Variable(entries['ControllerC3']['initial_value'], name = 'C_3')
                s0 = tf.Variable(entries['CompressiveElasticLimit']['initial_value'], \
                     name='CompressiveElasticLimit')
                sp = tf.Variable(entries['CompressiveStrength']['initial_value'], \
                     name='CompressiveStrength')
                sr = tf.Variable(entries['CompressiveResidualStrength']['initial_value'], \
                     name = 'CompressiveResidualStrength')
                sbi = tf.Variable(entries['BiaxialCompressiveStrength']['initial_value'], \
                     name = 'BiaxialCompressiveStrength')
                ep = tf.Variable(entries['StrainCompressiveStrength']['initial_value'], \
                     name = 'StrainCompressiveStrength')
                ft  = tf.Variable(entries['TensionStrength']['initial_value'], \
                     name='TensionStrength')
                gt  = tf.Variable(entries['TensionFractureEnergy']['initial_value'],\
                     name='TensionFractureEnergy')
            self.Variables= [s0, sp, sr, sbi, ep, ft, gt, c1, c2, c3]
            self.Vars4Print = ['Ep:  Strain Compressive Strength',\
                               'S0:  Compressive Elastic Limit', \
                               'Sp:  Compressive Strength', \
                               'Sr:  Compressive Residual Strength',\
                               'Sbi: Biaxial Compressive Strength', \
                               'Ft:  Tensile Strength',\
                               'Gt:  Fracture Energy Tension', \
                               'C1:  Bezier Controller 1',\
                               'C2:  Bezier Controller 2', \
                               'C3:  Bezier Controller 3']
            self.LearningRates = [                                       \
                entries['CompressiveElasticLimit']['learning_rate'],     \
                entries['CompressiveStrength']['learning_rate'],         \
                entries['CompressiveResidualStrength']['learning_rate'], \
                entries['BiaxialCompressiveStrength']['learning_rate'],  \
                entries['StrainCompressiveStrength']['learning_rate'],   \
                entries['TensionStrength']['learning_rate'],             \
                entries['TensionFractureEnergy']['learning_rate'],       \
                entries['ControllerC1']['learning_rate'],                \
                entries['ControllerC2']['learning_rate'],                \
                entries['ControllerC3']['learning_rate']                 \
            ]

        def ConstrainVariables(variables, variables_le):
            with tf.name_scope('ConstrainDamageVariables'):
                tol_s = 1e-10
                tol_e = 1e-10
                tol_g = 1e-10
                inf   = 1e+12
                
                ft = tf.clip_by_value(variables[5], tol_s, inf)
                gt = tf.clip_by_value(variables[6], tol_g, inf)

                sp  = tf.clip_by_value(variables[1], tol_s, inf)
                sbi = tf.clip_by_value(variables[3], tol_s, inf)
                s0  = tf.clip_by_value(variables[0], tol_s, tf.subtract(sp, tol_s))
                si = sp
                sj = sp
                sr = tf.clip_by_value(variables[2], tol_s, tf.multiply(sp,0.9))
                sk = tf.add(sr, tf.multiply(tf.subtract(sp,sr), variables[7]))
                su = sr

                e0 = tf.divide(s0, variables_le['E'])
                ei = tf.divide(si, variables_le['E'])
                ep = tf.clip_by_value(variables[4], tf.add(ei,tol_e), inf)

                alPHa = tf.multiply(2.0, tf.subtract(ep,tf.divide(sp,variables_le['E'])))

                ej = tf.add(ep, tf.multiply(variables[8],alPHa))
                ek = tf.add(ej, tf.multiply(alPHa, tf.subtract(1.0,variables[8])))
                
                er_t1 = tf.multiply(tf.subtract(ek, ej),tf.subtract(sk,sr))
                er_t2 = tf.divide(er_t1,tf.subtract(sj,sk))
                er_t3 = tf.add(ek,er_t2)
                er = tf.clip_by_value(er_t3, tf.add(ek,tol_e), inf)

                eu = tf.multiply(variables[9],er)

                ei = ModelVariables._petracca_bezier_update(e0, ei, ep)
                ej = ModelVariables._petracca_bezier_update(ep, ej, ek)
                er = ModelVariables._petracca_bezier_update(ek, er, eu)

            var_dict = {\
                'S0': s0, 'SI': si, 'SP': sp, 'SJ': sj, 'SR': sr, 'SK': sk, \
                'SU': su, 'SBI': sbi,\
                'E0': e0, 'EI': ei, 'EP': ep, 'EJ': ej, 'EK': ek, 'ER': er, \
                'EU': eu, \
                'FT': ft, 'GT': gt, 'C1': c1, 'C2': c2, 'C3': c3}
            return var_dict
    '''
       Additional Class to initialize the Tensorflow Summaries of the 
       TenExpoSoft and CompBezHardSoft Properties 
    '''               
    class ExpoSoftBezierHardSoftControlledSummary(object):
        def __init__(self, variables):
            with tf.name_scope('DamageParameters'):
                tf.summary.scalar('ControllerC1', \
                                  variables['C1'])
                tf.summary.scalar('ControllerC2', \
                                  variables['C2'])
                tf.summary.scalar('ControllerC3', \
                                  variables['C3'])                
                tf.summary.scalar('StrainCompressiveStrength', \
                                  variables['EP'])
                tf.summary.scalar('CompressiveElasticLimit', \
                                  variables['S0'])
                tf.summary.scalar('CompressiveStrength', \
                                  variables['SP'])
                tf.summary.scalar('CompressiveResidualStrength', \
                                  variables['SR'])
                tf.summary.scalar('BiaxialCompressiveStrength', \
                                  variables['SBI'])
                tf.summary.scalar('TensionStrength', \
                                  variables['FT'])
                tf.summary.scalar('TensionFractureEnergy', \
                                  variables['GT'])
        

                

                
                
                
   
               
            
   
