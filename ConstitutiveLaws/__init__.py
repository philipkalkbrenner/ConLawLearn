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
