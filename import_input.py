import numpy as np
import json
from random import sample
import sys

class TrainingInput(object):

    def __init__(self, input_settings):
        self.resize_fac = input_settings["resize_fac_stress"]
        self.tr_te_factor = input_settings["train_test_factor"] 

        self.input_names = input_settings["file_names"]
        
        self.strain_input_names = input_settings["strain_file_names"]
        
        self.strain_data = input_settings["strain_data"]
        self.data_collection = input_settings["data_collection"]
        
        if len(self.input_names) != len(self.strain_input_names):
            print("ERROR: Check the lists of the file names and the strain file names to import, ", "\n",\
            "their length must be equal!")
            sys.exit()


        if len(self.input_names) != 0:
            
            if self.data_collection == "manual":
                '''
                Method collects the data in the manual way.
                Input requierements:
                    - list of all the result files from the single ran analysis in the virtual lab
                    - manually made list of the corresponding last linear elastic result
                '''
                eps_le, eps_nl, sig_le, sig_nl = self.__manually_ordered_input_data(input_settings)

            elif self.data_collection == "automated":
                '''
                Method collects the data from a json file which is automatically generated from the virtual labs 
                and contains multiple analysis
                Input requierements:
                    - one single generated file from the virtual lab (name in the "strain_files" part of ModellSettings.json)
                    - a maximum norm to seperated expected linear elastic results sets from nonlinear ones, 
                      "splitting_criterion" in the ModelSettings.json 
                '''
                eps_le, eps_nl, sig_le, sig_nl = self.__automated_ordered_input_data(input_settings)
            elif self.data_collection == "for_print":
                pass

            else: 
                print("ERROR: No import method of the input data defined in the ModelSettings.json!", "\n", \
                        "either manual or automated is possible!")
                sys.exit()

            if self.data_collection == "for_print":
                '''
                Only defined to import the input strains for printing results
                '''
                eps_average, eps_boundary, sig_average = self.__import_strains_for_training(input_settings)
                self.GetAverageStrains  = np.asarray(eps_average, np.float32)
                self.GetBoundaryStrains = np.asarray(eps_boundary, np.float32)
                self.GetAverageStresses = np.asarray(sig_average, np.float32)
                eps_le = 0.0
                eps_nl = 0.0
                sig_le = 0.0
                sig_nl = 0.0

                #eps_le, eps_nl, sig_le, sig_nl = self.__automated_ordered_input_data(input_settings)


        
            self.GetStrainsLinearElastic     = np.asarray(eps_le, np.float32)
            self.GetStrainsNonlinear         = np.asarray(eps_nl, np.float32)

            self.GetStressesLinearElastic    = np.asarray(sig_le, np.float32) * self.resize_fac
            self.GetStressesNonlinear        = np.asarray(sig_nl, np.float32) * self.resize_fac

        elif self.strain_data == "unit_test":
            pass

        else:
            print("ERROR: No input defined in the ModelSettings.json!")
            sys.exit()

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
    ------------------------------------------------------------------------------------------------
    ------------------------------------------------------------------------------------------------
    '''       
    def __manually_ordered_input_data(self, input_settings):
        dam_start_step = input_settings["last_undamaged_step"]
        last_step_conv = input_settings["last_utilized_step"]

        eps_le = []
        sig_le = []
        eps_nl = []
        sig_nl = []

        for i in range(len(self.input_names)):
            with open(self.input_names[i]) as input_i:
                inp = json.load(input_i)
                stress  = inp["Mean_Value_of_CAUCHY_STRESS_VECTOR"]["Segment_1"]
                sig_le += stress[0:dam_start_step[i]]
                sig_nl += stress[dam_start_step[i]:last_step_conv[i]]

        if self.strain_data == "from_average":
            for i in range(len(self.input_names)):
                with open(self.input_names[i]) as input_i:
                    inp = json.load(input_i)
                        
                strains = inp["Mean_Value_of_GREEN_LAGRANGE_STRAIN_VECTOR"]["Segment_1"]
                eps_le += strains[0:dam_start_step[i]]
                eps_nl += strains[dam_start_step[i]:last_step_conv[i]]

        elif self.strain_data == "from_boundary":
            for i in range(len(self.strain_input_names)):
                with open(self.strain_input_names[i]) as strain_input_i:
                    inp = json.load(strain_input_i)
                strains = inp["Applied_Strains_at_Boundary"]
                eps_le += strains[0:dam_start_step[i]]
                eps_nl += strains[dam_start_step[i]:last_step_conv[i]]
                
        else:
            print("ERROR: No definition made in the ModelSettings.json, which strains should be utilized for the ML!", \
                "Either from_average or from_boundary is possible!")
            sys.exit()

        return eps_le, eps_nl, sig_le, sig_nl
    '''
    ------------------------------------------------------------------------------------------------
    '''   
    def __import_strains_for_training(self, input_settings):
        with open(self.input_names[0]) as input_data:
            input_data = json.load(input_data)

        stresses_average = input_data["Mean_Value_of_CAUCHY_STRESS_VECTOR"]
        strains_average = input_data["Mean_Value_of_GREEN_LAGRANGE_STRAIN_VECTOR"]
        strains_boundary = input_data["Values_of_BOUNDARY_STRAIN_VECTOR"]
        return strains_average, strains_boundary, stresses_average

    def __automated_ordered_input_data(self, input_settings):
        splitting_criterion = input_settings["splitting_criterion"]

        eps_le = []
        sig_le = []
        eps_nl = []
        sig_nl = []

        if len(self.input_names) != 1:
            print("ERROR: For the automated data collection only 1 input file is considered in this implementation!")
            sys.exit()

        with open(self.input_names[0]) as input_data:
            automated_data = json.load(input_data)

            stresses = automated_data["Mean_Value_of_CAUCHY_STRESS_VECTOR"]

        if self.strain_data == "from_average":
            strains = automated_data["Mean_Value_of_GREEN_LAGRANGE_STRAIN_VECTOR"]
        elif self.strain_data == "from_boundary":
            strains = automated_data["Values_of_BOUNDARY_STRAIN_VECTOR"]
        else:
            print("ERROR: No definition made in the ModelSettings.json, "\
                "which strains should be utilized for the ML!", \
                "Either from_average or from_boundary is possible!")
            sys.exit()

        strains_linear     = []
        stresses_linear    = []
        strains_nonlinear  = []
        stresses_nonlinear = []

        for i in range(len(strains)):
            norm = ((strains[i][0]**2) + (strains[i][1]**2) + (strains[i][2]**2))**0.5
    
            if norm < splitting_criterion:
                strains_linear.append(strains[i])
                stresses_linear.append(stresses[i])
            else:
                strains_nonlinear.append(strains[i])
                stresses_nonlinear.append(stresses[i])
        if len(strains_linear) == 0:
            print("ERROR: The criterion to differ the linear range from the nonlinear "\
                    "one is too small. The array of strains in the linear elastic range is empty."\
                    "Decrease the criteria in the ModelSettings.json!")
            sys.exit()
        if len(strains_nonlinear) == 0:
            print("ERROR: The criterion to differ the linear range from the nonlinear "\
                "one is too large or no nonlinear values exist. The array of strains" \
                " in the nonlinear range is empty."\
                " Increase the criteria in the ModelSettings.json or rediscuss your input data!")
            sys.exit()

        eps_le = strains_linear
        eps_nl = strains_nonlinear
        sig_le = stresses_linear 
        sig_nl = stresses_nonlinear

        return eps_le, eps_nl, sig_le, sig_nl
    '''
    ------------------------------------------------------------------------------------------------
    '''       
                
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
        
        
        
