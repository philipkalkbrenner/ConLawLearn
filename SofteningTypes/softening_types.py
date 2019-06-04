import tensorflow as tf

class SofteningType(object):
    class ExponentialSoftening(object):
        
        def GetDamageThreshold(tau, f):
            r, r0 =  SofteningType._Damage_Threshold(tau, f)
            return r
            
        def GetDamageVariable(tau, e, f, g):
            r, r0 = SofteningType._Damage_Threshold(tau, f)
            hd = SofteningType._Discrete_Softening_Parameter_Exp_Soft(e, f, g)
            d = SofteningType._Damage_Variable_Exp_Soft(r, r0, hd)            
            return d


    class ParabolicHardeningExponentialSoftening(object):
        def GetDamageThreshold(tau,r0):
            (r, _) = SofteningType._Damage_Threshold(tau, r0)
            return r

        def GetDamageVariable(tau, e, f, g, r0, re, rp):
            with tf. name_scope('D_ParHardExpSoft'):
                (r, _) = SofteningType._Damage_Threshold(tau, r0)
                hd = SofteningType._Discrete_Softening_Parameter_Par_Hard_Exp_Soft\
                     (r0, re, rp, e, f, g)
                d_hard = SofteningType._Damage_Variable_Par_Hard(r, r0, re, rp)
                d_soft = SofteningType._Damage_Variable_Par_Hard_Exp_Soft(r, re, rp, hd)
            d = SofteningType._Damage_Variable_Selection_Par_Hard_Exp_Soft\
                                (r, d_hard, d_soft, r0, rp)
            return d

    class BezierHardeningSoftening(object):
        
        def GetDamageThreshold(tau,s0):
            (r, _) = SofteningType._Damage_Threshold(tau, s0)
            return r
        def GetStrainLikeCounterPart(r,e):
            Xi = SofteningType._StrainLikeCounterPart(r,e)
            return Xi
        
        def GetBezierFactors(r, e, e0, ei, ep, ej, ek, er, eu, s0, si, sp, sj, sk, sr, su):
            (A, B, C, D, t, BezSum) = SofteningType._GetBezierFactors(r, e, e0, ei, ep, ej, ek, er, eu, \
                                                                            s0, si, sp, sj, sk, sr, su)
            return (A, B, C, D, t, BezSum)
    
        def GetDamageVariable(tau, e, e0, ei, ep, ej, ek, er, eu, \
                              s0, si, sp, sj, sk, sr, su):
            with tf.name_scope('BezHardSoft'):
                (r, _) = SofteningType._Damage_Threshold(tau, s0)

                '''
                Xi = SofteningType._StrainLikeCounterPart(r,e)

                d_lin = tf.zeros_like(r)
                d_resi = tf.add(tf.zeros_like(r), tf.subtract(1.0,tf.divide(su,r)))
                d_hard, d_soft1, d_soft2 = SofteningType._Damage_Variable_Bezier_Conditioning_Newer\
                         (r, e, e0, ei, ep, ej, ek, er, eu, s0, si, sp, sj, sk, sr, su)
                c_lin, c_hard, c_soft1, c_soft2, c_resi = \
                        SofteningType._GetConditionsIntervals(Xi, e0, ep, ek, eu)
                d_newer = tf.where(c_lin, d_lin, \
                         tf.where(c_hard, d_hard, \
                                  tf.where(c_soft1, d_soft1,\
                                           tf.where(c_soft2, d_soft2, d_resi)\
                        )))  
                        '''

                d_new = SofteningType._Damage_Variable_Bezier_Conditioning_New\
                         (r, e, e0, ei, ep, ej, ek, er, eu, s0, si, sp, sj, sk, sr, su)
                
                Bezier = SofteningType._Damage_Variable_Bezier_Conditioning\
                         (r, e, e0, ei, ep, ej, ek, er, eu, s0, si, sp, sj, sk, sr, su)
            #d_old = SofteningType._Damage_Variable_Bezier(Bezier, r)
            
            return d_new        
             
    '''
-------------------------------------------------------------------------------
    '''
    def _StrainLikeCounterPart(r,e):
        xi = SofteningType.__get_strain_like_counterpart(r,e)
        return xi

    def _Damage_Threshold(tau, f):
        with tf.name_scope('Computation_A'):
            r0 = tf.fill(tf.shape(tau), f)
            cond = tf.greater(tau, r0)
        r = tf.where(cond, tau, r0, name='DamageThreshold')
        return (r, r0)

        
    def _Discrete_Softening_Parameter_Exp_Soft(e, f, gf):
        with tf.name_scope('Hd_Exp_Soft'):
            with tf.name_scope('Computation_B'):
                h_temp1 = tf.multiply(tf.divide(tf.multiply(2.0, e), \
                                                tf.square(f)),gf)
                h_temp2 = tf.subtract(h_temp1, 1.0)
            h =tf.divide(1.0, h_temp2, name='HdPos')
        return h

    def _Discrete_Softening_Parameter_Par_Hard_Exp_Soft(r0, re, rp, e, f, g):
        with tf.name_scope('Hd_Par_Hard_Exp_Soft'):
            ad_snake = SofteningType.__get_Ad_snake_para_hard(rp, r0, re)
            temp1 = tf.multiply(tf.divide(tf.multiply(2.0, e), \
                                              tf.square(f)), g)
            temp2 = tf.divide(rp, re)
            temp3 = tf.subtract(tf.subtract(temp1, ad_snake),temp2)
        h = tf.divide(1.0,temp3, name='Hd_Para_Hard_Exp_Soft')
        return h

    def _Damage_Variable_Exp_Soft(r, r0, hd):
        with tf.name_scope('Comp_D_ExpoSoft'):
            d_temp1 = tf.divide(r0,r, name = "temp1")
            d_temp2 = tf.multiply(2.0, tf.multiply(hd, \
                                    tf.divide(tf.subtract(r0,r), r0)), name = "temp2")
            d_temp3 = tf.exp(d_temp2, name = "temp3")
            d_temp4 = tf.multiply(d_temp1, d_temp3, name = "temp4")
        d = tf.subtract(1.0,d_temp4, name="Damagevariable")
        return d

    def _Damage_Variable_Par_Hard(r, r0, re, rp):
        with tf.name_scope('D_ParHard'):
            ad = SofteningType.__get_Ad_para_hard_exp_soft(rp, re)
            d_temp1 = tf.square(tf.divide(tf.subtract(r, r0), \
                                    tf.subtract(rp, r0)))
        
            d_temp2 = tf.multiply(tf.divide(re, r), ad)
        d = tf.multiply(d_temp1, d_temp2, name='D_ParHard')
        return d

    def _Damage_Variable_Par_Hard_Exp_Soft(r, re, rp, hd):
        with tf.name_scope('D_ParHardExpSoft'):
            d_temp1 = tf.multiply(\
                tf.multiply(2.0, hd),tf.divide(tf.subtract(rp, r), re))
            d_temp2 = tf.exp(d_temp1)
            d_temp3 = tf.multiply(d_temp2, tf.divide(re, r))
        d = tf.subtract(1.0, d_temp3, name='D_PHExpSoft')
        return d

    def _Damage_Variable_Selection_Par_Hard_Exp_Soft(r, d_hard, d_soft, r0, rp):
        with tf.name_scope('D_ParHardExpSoft_Selection'):
            R0 = tf.fill(tf.shape(r), r0)
            Rp = tf.fill(tf.shape(r), rp)
            cond1 = tf.greater(r, Rp)
            cond2 = tf.greater(r, R0)
        d = tf.where(cond1, d_soft, tf.where(cond2, d_hard, tf.zeros_like(r))\
            ,name='D_ParHardExpSoft')
        return d

    def _Damage_Variable_Bezier_Conditioning(r, e, e0, ei, ep, ej, ek, er, eu, \
                                             s0, si, sp, sj, sk, sr, su):
        Xi = SofteningType.__get_strain_like_counterpart(r,e)

        B_lin_temp   = r
        B_hard_temp  = SofteningType.__call_bezier_function(Xi,e0,ei,ep,s0,si,sp)
        B_soft1_temp = SofteningType.__call_bezier_function(Xi,ep,ej,ek,sp,sj,sk)
        B_soft2_temp = SofteningType.__call_bezier_function(Xi,ek,er,eu,sk,sr,su)
        B_resi_temp  = tf.fill(tf.shape(Xi),su)
        
        c_lin, c_hard, c_soft1, c_soft2, c_resi = \
               SofteningType.__get_conditions_intervals(Xi, e0, ep, ek, eu)
        
        B_lin   = SofteningType.__apply_condition_to_bezier(B_lin_temp, c_lin)
        B_hard  = SofteningType.__apply_condition_to_bezier(B_hard_temp, c_hard)
        B_soft1 = SofteningType.__apply_condition_to_bezier(B_soft1_temp, c_soft1)
        B_soft2 = SofteningType.__apply_condition_to_bezier(B_soft2_temp, c_soft2)
        B_resi  = SofteningType.__apply_condition_to_bezier(B_resi_temp, c_resi)

        B_tot = tf.add(tf.add(tf.add(B_lin, B_hard),tf.add(B_soft1, B_soft2)), B_resi)
        return B_tot

    def _Damage_Variable_Bezier(Bez, r):
        d = tf.subtract(1.0, tf.divide(Bez, r), name='D_BezierHardSoft')
        return d

    def _Damage_Variable_Bezier_Conditioning_New(r, e, e0, ei, ep, ej, ek, er, eu, \
                                             s0, si, sp, sj, sk, sr, su):
        with tf.name_scope("StrainLikePart"):
            Xi = SofteningType.__get_strain_like_counterpart(r,e)
        with tf.name_scope("BezConds"):
            c_lin, c_hard, c_soft1, c_soft2, c_resi = \
                SofteningType.__get_conditions_intervals(Xi, e0, ep, ek, eu)

            Xi_hard  = tf.where(c_hard, Xi, tf.zeros_like(Xi))
            Xi_soft1 = tf.where(c_soft1, Xi, tf.zeros_like(Xi))
            Xi_soft2 = tf.where(c_soft2, Xi, tf.zeros_like(Xi))


        with tf.name_scope("LinElaRange"):
            Bez_lin   = r
            d_lin   = SofteningType.__damage_variable_bezier(Bez_lin, r)
        with tf.name_scope("BezHardgRange"):
            Bez_hard  = SofteningType.__call_bezier_function(Xi_hard,e0,ei,ep,s0,si,sp)
            d_hard  = SofteningType.__damage_variable_bezier(Bez_hard, r)
        with tf.name_scope("BezSoftRange1"):
            Bez_soft1 = SofteningType.__call_bezier_function(Xi_soft1,ep,ej,ek,sp,sj,sk)
            d_soft1 = SofteningType.__damage_variable_bezier(Bez_soft1, r)
        with tf.name_scope("BezSoftRange2"):
            Bez_soft2 = SofteningType.__call_bezier_function(Xi_soft2,ek,er,eu,sk,sr,su)
            d_soft2 = SofteningType.__damage_variable_bezier(Bez_soft2, r)
        with tf.name_scope("ResiRange"):
            Bez_resi  = tf.fill(tf.shape(Xi),su)
            d_resi  = SofteningType.__damage_variable_bezier(Bez_resi, r)
        '''
        d_lin   = SofteningType.__damage_variable_bezier(Bez_lin, r)
        d_hard  = SofteningType.__damage_variable_bezier(Bez_hard, r)
        d_soft1 = SofteningType.__damage_variable_bezier(Bez_soft1, r)
        d_soft2 = SofteningType.__damage_variable_bezier(Bez_soft2, r)
        d_resi  = SofteningType.__damage_variable_bezier(Bez_resi, r)
        '''
        with tf.name_scope("BezierSelect"):
            d_tot = tf.where(c_lin, d_lin, \
                         tf.where(c_hard, d_hard, \
                                  tf.where(c_soft1, d_soft1,\
                                           tf.where(c_soft2, d_soft2, d_resi)\
                        )))    
        return d_tot

    def _Damage_Variable_Bezier_Conditioning_Newer(r, e, e0, ei, ep, ej, ek, er, eu, \
                                             s0, si, sp, sj, sk, sr, su):
        Xi = SofteningType.__get_strain_like_counterpart(r,e)

        Bez_hard  = SofteningType.__call_bezier_function(Xi,e0,ei,ep,s0,si,sp)
        Bez_soft1 = SofteningType.__call_bezier_function(Xi,ep,ej,ek,sp,sj,sk)
        Bez_soft2 = SofteningType.__call_bezier_function(Xi,ek,er,eu,sk,sr,su)
        
        d_hard  = SofteningType.__damage_variable_bezier(Bez_hard, r)
        d_soft1 = SofteningType.__damage_variable_bezier(Bez_soft1, r)
        d_soft2 = SofteningType.__damage_variable_bezier(Bez_soft2, r)

        return d_hard, d_soft1, d_soft2


    def _GetBezierFactors(r, e, e0, ei, ep, ej, ek, er, eu, s0, si , sp, sj, sk, sr, su):
        Xi = SofteningType.__get_strain_like_counterpart(r,e)

        def getA(Xi, x1,x2,x3):
            A = tf.zeros_like(Xi)
            A = tf.add(A, tf.subtract(tf.add(x1, x3), tf.multiply(2.0, x2)))
            return A
        def getB(Xi,x1,x2):
            B = tf.zeros_like(Xi)
            B = tf.add(B,tf.multiply(2.0, tf.subtract(x2, x1)))
            return B 
        def getC(Xi,x1):
            C = tf.subtract(x1, Xi)
            return C 
        def getD(A,B,C):
            D = tf.subtract(tf.square(B),tf.multiply(tf.multiply(4.0,A), C))
            return D 
        def gett(A,B,D):
            t = tf.divide(tf.subtract(tf.sqrt(D), B), \
                              tf.multiply(2.0,A))
            return t
        def getBezSum(t, y1, y2, y3):
            Bez_1 = tf.multiply(tf.add(tf.subtract(y1, \
                                       tf.multiply(2.0, y2)), y3), \
                                       tf.pow(t,2.0))
            Bez_2 = tf.multiply(tf.multiply(2.0,t),tf.subtract(y2,y1))
            BezSum = tf.add(tf.add(Bez_1,Bez_2),y1)
            return BezSum
            

        A_lin = tf.add(tf.zeros_like(Xi),666)
        B_lin = tf.add(tf.zeros_like(Xi),666)
        C_lin = tf.add(tf.zeros_like(Xi),666)
        D_lin = tf.add(tf.zeros_like(Xi),666)
        t_lin = tf.add(tf.zeros_like(Xi),666)
        Bez_sum_lin = tf.add(tf.zeros_like(Xi),666)

        A_hard = getA(Xi,e0, ei, ep)
        B_hard = getB(Xi, e0, ei)
        C_hard = getC(Xi, e0)
        D_hard = getD(A_hard, B_hard, C_hard)
        t_hard = gett(A_hard, B_hard, D_hard)
        Bez_sum_hard = getBezSum(t_hard, s0, si, sp)

        A_soft1 = getA(Xi, ep, ej, ek)
        B_soft1 = getB(Xi, ep, ej)
        C_soft1 = getC(Xi, ep)
        D_soft1 = getD(A_soft1, B_soft1, C_soft1)
        t_soft1 = gett(A_soft1, B_soft1, D_soft1)
        Bez_sum_soft1 = getBezSum(t_soft1, sp, sj, sk)

        A_soft2 = getA(Xi, ek, er, eu)
        B_soft2 = getB(Xi, ek, er)
        C_soft2 = getC(Xi, ek)
        D_soft2 = getD(A_soft2, B_soft2, C_soft2)
        t_soft2 = gett(A_soft2, B_soft2, D_soft2)
        Bez_sum_soft2 = getBezSum(t_soft2, sk, sr, su)

        A_resi = tf.add(tf.zeros_like(Xi),777)
        B_resi = tf.add(tf.zeros_like(Xi),777)
        C_resi = tf.add(tf.zeros_like(Xi),777)
        D_resi = tf.add(tf.zeros_like(Xi),777)
        t_resi = tf.add(tf.zeros_like(Xi),777)
        Bez_sum_resi = tf.add(tf.zeros_like(Xi),777)

        c_lin, c_hard, c_soft1, c_soft2, c_resi = \
               SofteningType.__get_conditions_intervals(Xi, e0, ep, ek, eu)
        
        A = tf.where(c_lin, A_lin, \
                         tf.where(c_hard, A_hard, \
                                  tf.where(c_soft1, A_soft1,\
                                           tf.where(c_soft2, A_soft2, A_resi))))
        B = tf.where(c_lin, B_lin, \
                         tf.where(c_hard, B_hard, \
                                  tf.where(c_soft1, B_soft1,\
                                           tf.where(c_soft2, B_soft2, B_resi))))
        C = tf.where(c_lin, C_lin, \
                         tf.where(c_hard, C_hard, \
                                  tf.where(c_soft1, C_soft1,\
                                           tf.where(c_soft2, C_soft2, C_resi))))
        D = tf.where(c_lin, D_lin, \
                         tf.where(c_hard, D_hard, \
                                  tf.where(c_soft1, D_soft1,\
                                           tf.where(c_soft2, D_soft2, D_resi))))
        t = tf.where(c_lin, t_lin, \
                         tf.where(c_hard, t_hard, \
                                  tf.where(c_soft1, t_soft1,\
                                           tf.where(c_soft2, t_soft2, t_resi))))

        BezSum = tf.where(c_lin, Bez_sum_lin, \
                         tf.where(c_hard, Bez_sum_hard, \
                                  tf.where(c_soft1, Bez_sum_soft1,\
                                           tf.where(c_soft2, Bez_sum_soft2, Bez_sum_resi))))
        
        return (A,B,C,D,t, BezSum)


    def _GetConditionsIntervals(Xi, e0, ep, ek, eu):
        with tf.name_scope('Conditions'):
            c_lin, c_hard, c_soft1, c_soft2, c_resi = \
               SofteningType.__get_conditions_intervals(Xi, e0, ep, ek, eu)
        return (c_lin, c_hard, c_soft1, c_soft2, c_resi)
        
    '''
        Functions for Parabolic Hardening and Exponential Softening:
        -----------------------------------------------------------------------
    '''

    def __get_Ad_para_hard_exp_soft(rp,re):
        with tf.name_scope('Ad'):
            ad = tf.divide(tf.subtract(rp, re), re)
        return ad

    def __get_Ad_snake_para_hard(rp,r0,re):
        with tf.name_scope('Ad_snake'):
            ad = SofteningType.__get_Ad_para_hard_exp_soft(rp,re)
            temp1 = tf.multiply(2.0, tf.pow(r0,3))
            temp2 = tf.multiply(3.0, tf.multiply(rp, tf.square(r0)))
            temp3 = tf.pow(rp, 3)
            temp4 = tf.add(tf.subtract(temp3, temp2), temp1)
            temp5 = tf.multiply(ad, temp4)
            temp6 = tf.square(tf.subtract(rp, r0))
            temp7 = tf.multiply(3.0, tf.multiply(re, temp6))
        ad_snake = tf.divide(temp5, temp7, name="Ad_snake")
        return ad_snake

    def __get_Bd_para_hard(rp,r0,re):
        with tf.name_scop('Bd'):
            ad = SofteningType.__get_Ad_para_hard_exp_soft(rp,re)
            temp1 = tf.subtract(rp,r0)
            temp2 = tf.subtract(tf.multiply(3.0, rp), tf.multiply(2.0, \
                                                        tf.multiply(ad,re)))
            temp3 = tf.add(temp2, tf.multiply(3.0, r0))
            temp4 = tf.multiply(temp3, temp1)
            temp5 = tf.multiply(3.0, tf.square(re))
        bd = tf.divide(temp4, temp5, name='Bd')
        return bd
        
    '''
        Functions for Bezier Hardening and Softening:
        -----------------------------------------------------------------------
    '''    

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


    def __get_strain_like_counterpart(r,e):
        XI = tf.divide(r,e,name='StrainLikeCounter')
        return XI

    def __get_conditions_intervals(Xi, e0, ep, ek, eu):
        c_lin   = tf.less_equal(Xi,e0)
        c_hard  = tf.logical_and(tf.greater(Xi,e0), tf.less_equal(Xi,ep))
        c_soft1 = tf.logical_and(tf.greater(Xi,ep), tf.less_equal(Xi,ek))
        c_soft2 = tf.logical_and(tf.greater(Xi,ek), tf.less_equal(Xi,eu))
        c_resi  = tf.greater(Xi,eu)
        #for Masking:
        #c_lin.set_shape([None])
        #c_hard.set_shape([None])
        #c_soft1.set_shape([None])
        #c_soft2.set_shape([None])
        #c_resi.set_shape([None])
        return (c_lin, c_hard, c_soft1, c_soft2, c_resi)

    def __apply_condition_to_bezier(Bez, cond):
        with tf.name_scope('Apply_Condition'):
            Bez_temp = tf.where(cond, Bez, tf.zeros_like(Bez))
        return Bez_temp 

    def __compute_bezier_damage_variable(Bez, r):
        d = tf.subtract(1.0, tf.divide(Bez, r), name='D_BezierHardeSoft')
        return d

    def __damage_variable_bezier(Bez, r):
        d = tf.subtract(1.0, tf.divide(Bez, r), name='DamageTemp')
        return d
        
    
    
    
            

        
    
    
