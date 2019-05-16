import numpy as np
import json
from random import sample

class TrainingInput(object):

    def __init__(self, input_settings):
        self.resize_fac = input_settings["resize_fac_stress"]
        self.tr_te_factor = input_settings["train_test_factor"] 

        self.input_names = input_settings["file_names"]

        if len(self.input_names) != 0:
            dam_start_step = input_settings["last_undamaged_step"]
            last_step_conv = input_settings["last_converged_step"]
        
            eps_le = []
            sig_le = []
            eps_nl = []
            sig_nl = []
            damage_pos = []
            damage_neg = []
            sig_pos_le = []
            sig_pos_nl = []
            sig_neg_le = []
            sig_neg_nl = []
            sig_eff_pos_le = []
            sig_eff_pos_nl = []
            sig_eff_neg_le = []
            sig_eff_neg_nl = []

            for i in range(len(self.input_names)):
                with open(self.input_names[i]) as input_i:
                    inp = json.load(input_i)
                    strains = inp["Mean_Value_of_GREEN_LAGRANGE_STRAIN_VECTOR"]["Segment_1"]
                    eps_le += strains[0:dam_start_step[i]]
                    eps_nl += strains[dam_start_step[i]:last_step_conv[i]]
                
                    stress  = inp["Mean_Value_of_CAUCHY_STRESS_VECTOR"]["Segment_1"]
                    sig_le += stress[0:dam_start_step[i]]
                    sig_nl += stress[dam_start_step[i]:last_step_conv[i]]
                
                    dam_pos = inp["Mean_Value_of_DAMAGE_TENSION"]["Segment_1"]
                    dam_neg = inp["Mean_Value_of_DAMAGE_COMPRESSION"]["Segment_1"]
                    damage_pos += dam_pos[dam_start_step[i]:last_step_conv[i]]
                    damage_neg += dam_neg[dam_start_step[i]:last_step_conv[i]]
                
                    stress_pos = inp["Mean_Value_of_TENSION_STRESS_VECTOR"]["Segment_1"]
                    stress_neg = inp["Mean_Value_of_COMPRESSION_STRESS_VECTOR"]["Segment_1"]
                    sig_pos_le += stress_pos[0:dam_start_step[i]]
                    sig_pos_nl += stress_pos[dam_start_step[i]:last_step_conv[i]]
                    sig_neg_le += stress_neg[0:dam_start_step[i]]
                    sig_neg_nl += stress_neg[dam_start_step[i]:last_step_conv[i]]
                
            eps_le = np.asarray(eps_le, np.float32)
            sig_le = np.asarray(sig_le, np.float32)
            eps_nl = np.asarray(eps_nl, np.float32)
            sig_nl = np.asarray(sig_nl, np.float32)
            damage_pos = np.asarray(damage_pos, np.float32)
            damage_neg = np.asarray(damage_neg, np.float32)
            sig_pos_le = np.asarray(sig_pos_le)
            sig_pos_nl = np.asarray(sig_pos_nl)
            sig_neg_le = np.asarray(sig_neg_le)
            sig_neg_nl = np.asarray(sig_neg_nl)
        
            self.GetStrainsLinearElastic     = eps_le
            self.GetStrainsNonlinear         = eps_nl
            self.GetStressesLinearElastic    = sig_le * self.resize_fac
            self.GetStressesNonlinear        = sig_nl * self.resize_fac
            self.GetPositiveDamage           = damage_pos
            self.GetNegativeDamage           = damage_neg
            self.GetPositeStressLinear       = sig_pos_le * self.resize_fac
            self.GetNegativeStressLinear     = sig_neg_le * self.resize_fac
            self.GetPositeStressNonlinear    = sig_pos_nl * self.resize_fac
            self.GetNegativeStressNonlinear  = sig_neg_nl * self.resize_fac

    def SplitTrainingAndTesting(self, eps, sig):
        l = eps.shape[0]
        f = round(self.tr_te_factor*l)
        ind = sample(range(l), f)
        eps_train = eps[ind]
        eps_test  = np.delete(eps, ind, 0)
        sig_train = sig[ind]
        sig_test  = np.delete(sig, ind, 0)
        return eps_train, eps_test, sig_train, sig_test

    def SplitTrainingAndTesting3Entries(self, eps, sig_pos, sig_neg):
        l = eps.shape[0]
        f = round(self.tr_te_factor*l)
        ind = sample(range(l), f)
        eps_train = eps[ind]
        eps_test  = np.delete(eps, ind, 0)
        sig_pos_train = sig_pos[ind]
        sig_pos_test  = np.delete(sig_pos, ind, 0)
        sig_neg_train = sig_pos[ind]
        sig_neg_test  = np.delete(sig_pos, ind, 0)
        return eps_train, eps_test, sig_pos_train, sig_pos_test, sig_neg_train, sig_neg_test

    def SplitTrainingAndTesting5Entries(self, eps, sig_pos, sig_neg, sig_eff_pos, sig_eff_neg):
        l = eps.shape[0]
        f = round(self.tr_te_factor*l)
        ind = sample(range(l), f)
        eps_train = eps[ind]
        eps_test  = np.delete(eps, ind, 0)
        sig_pos_train = sig_pos[ind]
        sig_pos_test  = np.delete(sig_pos, ind, 0)
        sig_neg_train = sig_pos[ind]
        sig_neg_test  = np.delete(sig_pos, ind, 0)
        sig_eff_pos_train = sig_eff_pos[ind]
        sig_eff_pos_test  = np.delete(sig_eff_pos, ind, 0)
        sig_eff_neg_train = sig_eff_neg[ind]
        sig_eff_neg_test  = np.delete(sig_eff_neg, ind, 0)
        return eps_train, eps_test, sig_pos_train, sig_pos_test, sig_neg_train, sig_neg_test, \
               sig_eff_pos_train, sig_eff_pos_test, sig_eff_neg_train, sig_eff_neg_test

    def GetStrainsAndStressesForPostPlot(self):
        length_input_set = len(self.input_names)
        file_to_take = np.random.randint(1, length_input_set+1)
        with open(self.input_names[file_to_take]) as input_i:
                inp = json.load(input_i)
        epsilon = inp["Mean_Value_of_GREEN_LAGRANGE_STRAIN_VECTOR"]["Segment_1"]
        sigma   = inp["Mean_Value_of_CAUCHY_STRESS_TENSOR"]["Segment_1"]
        return epsilon,sigma
        
        
                
'''        
        
    def GetStrainsLinearElastic(input_settings):
        input_names = input_settings["linear_elastic"]["file_names"]
        eps = []
        for i in range(len(input_names)):
            with open(input_names[i]) as input_i:
                inp = json.load(input_i)
                eps += inp["Mean_Value_of_GREEN_LAGRANGE_STRAIN_VECTOR"]["Segment_1"]

        eps = np.asarray(eps)
        return eps

    def GetStressesLinearElastic(input_settings):
        input_names = input_settings["linear_elastic"]["file_names"]
        resize_fac = input_settings["resize_fac_stress"]
        sig = []
        for i in range(len(input_names)):
            with open(input_names[i]) as input_i:
                inp = json.load(input_i)
                sig += inp["Mean_Value_of_CAUCHY_STRESS_VECTOR"]["Segment_1"]

        sig = np.asarray(sig) * resize_fac
        return sig

    def GetStrainsNonlinear(input_settings):
        input_names = input_settings["non_linear"]["file_names"]
        dam_start_step = input_settings["non_linear"]["last_undamaged_step"]
        last_step_conv = input_settings["non_linear"]["last_converged_step"]
        eps = []
        for i in range(len(input_names)):
            with open(input_names[i]) as input_i:
                inp = json.load(input_i)
                strains = inp["Mean_Value_of_GREEN_LAGRANGE_STRAIN_VECTOR"]["Segment_1"]
                eps += strains[dam_start_step[i]:last_step_conv[i]]

        eps = np.asarray(eps)
        return eps

    def GetStressesNonlinear(input_settings):
        input_names = input_settings["non_linear"]["file_names"]
        dam_start_step = input_settings["non_linear"]["last_undamaged_step"]
        last_step_conv = input_settings["non_linear"]["last_converged_step"]
        resize_fac = input_settings["resize_fac_stress"]
        sig = []
        for i in range(len(input_names)):
            with open(input_names[i]) as input_i:
                inp = json.load(input_i)
                stress = inp["Mean_Value_of_CAUCHY_STRESS_VECTOR"]["Segment_1"]
                sig += stress[dam_start_step[i]:last_step_conv[i]]

        sig = np.asarray(sig) * resize_fac
        return sig

'''    
        
        
        
