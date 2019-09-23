'''
Linear Laws
'''
# Linear elastic plane stress:
from ConLawLearn.ConstitutiveLaws.linear_elastic_plane_stress import LinearElasticPlaneStress
# Linear elastic plane strain:
from ConLawLearn.ConstitutiveLaws.linear_elastic_plane_strain import LinearElasticPlaneStrain

'''
Damage Laws with Petracca modified Lubliner Yield surface
'''
# exponential softening (tension) & bezier hardening and softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_petracca_exposoft_neg_petracca_bezierhardsoft import PosPetraccaExpoSoftNegPetraccaBezierHardSoft
# exponential softening (tension) & parabolic hardening and exponential softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_petracca_exposoft_neg_petracca_parhardexposoft import PosPetraccaExpoSoftNegPetraccaParHardExpoSoft
# exponential softening (tension) & exponential softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_petracca_exposoft_neg_petracca_exposoft import PosPetraccaExpoSoftNegPetraccaExpoSoft
# linear softening (tension) & linear softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_petracca_linsoft_neg_petracca_linsoft import PosPetraccaLinSoftNegPetraccaLinSoft


'''
Damage Laws with Rankine Yield surface
'''
# linear softening (tension) & linear softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_rankine_linsoft_neg_rankine_linsoft import PosRankineLinSoftNegRankineLinSoft


'''
Damage with Petracca modified Lubliner Yield surface in Tension and
            Lubliner Yield surface in Compression
'''
# linear softening (tension) & linear softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_petracca_linsoft_neg_lubliner_linsoft import PosPetraccaLinSoftNegLublinerLinSoft
# exponential softening (tension) & exponential softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_petracca_exposoft_neg_lubliner_exposoft import PosPetraccaExpoSoftNegLublinerExpoSoft
# exponential softening (tension) & parabolic hardening and exponential softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_petracca_exposoft_neg_lubliner_parhardexposoft import PosPetraccaExpoSoftNegLublinerParHardExpoSoft
# exponential softening (tension) & bezier hardening and softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_petracca_exposoft_neg_lubliner_bezierhardsoft import PosPetraccaExpoSoftNegLublinerBezierHardSoft


'''
Damage with Petracca modified Lubliner Yield surface in Tension and
            Drucker Prager Yield surface in Compression
'''
# linear softening (tension) & linear softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_petracca_linsoft_neg_drucprag_linsoft import PosPetraccaLinSoftNegDrucPragLinSoft


'''
Damage with Petracca modified Lubliner Yield surface in Tension and
            Rankine Yield surface in Compression
'''
# linear softening (tension) & linear softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_petracca_linsoft_neg_rankine_linsoft import PosPetraccaLinSoftNegRankineLinSoft


'''
Damage with Rankine Yield surface in Tension and
            Petracca modified Lubliner Yield surface in Compression
'''
# linear softening (tension) & linear softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_rankine_linsoft_neg_petracca_linsoft import PosRankineLinSoftNegPetraccaLinSoft


'''
Damage with Rankine Yield surface in Tension and
            Lubliner Yield surface in Compression
'''
# linear softening (tension) & linear softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_rankine_linsoft_neg_lubliner_linsoft import PosRankineLinSoftNegLublinerLinSoft
# exponential softening (tension) & exponential softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_rankine_exposoft_neg_petracca_exposoft import PosRankineExpoSoftNegPetraccaExpoSoft
# exponential softening (tension) & parabolic hardening and exponential softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_rankine_exposoft_neg_petracca_parhardexposoft import PosRankineExpoSoftNegPetraccaParHardExpoSoft
# exponential softening (tension) & bezier hardening and softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_rankine_exposoft_neg_petracca_bezierhardsoft import PosRankineExpoSoftNegPetraccaBezierHardSoft

'''
Damage with Rankine Yield surface in Tension and
            Drucker Prager Yield surface in Compression
'''
# linear softening (tension) & linear softening (compression)
from ConLawLearn.ConstitutiveLaws.damage_pos_rankine_linsoft_neg_drucprag_linsoft import PosRankineLinSoftNegDrucPragLinSoft











'''
Damage Laws with Lubliner Yield surface
'''
# exponential softneing (tension) & bezier hardening and softening (compression)
#from ConLawLearn.ConstitutiveLaws.damage_pos_lubliner_exposoft_neg_lubliner_bezierhardsoft import PosLublinerExpoSoftNegLublinerBezierHardSoft
# exponential softneing (tension) & parabolic hardening and exponential softening (compression)
#from ConLawLearn.ConstitutiveLaws.damage_pos_lubliner_exposoft_neg_lubliner_parhardexposoft import PosLublinerExpoSoftNegLublinerParHardExpoSoft
# exponential softneing (tension) & exponential softening (compression)
#from ConLawLearn.ConstitutiveLaws.damage_pos_lubliner_exposoft_neg_lubliner_exposoft import PosLublinerExpoSoftNegLublinerExpoSoft

'''
Damage Laws with Drucker Prager Yield surface
'''
# exponential softneing (tension) & bezier hardening and softening (compression)
#from ConLawLearn.ConstitutiveLaws.damage_pos_drucprag_exposoft_neg_drucprag_bezierhardsoft import PosDrucPragExpoSoftNegDrucPragBezierHardSoft
# exponential softneing (tension) & parabolic hardening and exponential softening (compression)
#from ConLawLearn.ConstitutiveLaws.damage_pos_drucprag_exposoft_neg_drucprag_parhardexposoft import PosDrucPragExpoSoftNegDrucPragParHardExpoSoft
# exponential softneing (tension) & exponential softening (compression)
#from ConLawLearn.ConstitutiveLaws.damage_pos_drucprag_exposoft_neg_drucprag_exposoft import PosDrucPragExpoSoftNegDrucPragExpoSoft
