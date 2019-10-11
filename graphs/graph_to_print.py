import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import ConLawLearn as ConLawL

class GraphToPrint(object):
    def __init__(self, model_settings, initial_variable_values):
        self.model_settings = model_settings
        self.init_var_values = initial_variable_values

        self.input_settings = self.model_settings["input_settings"]
        self.post_settings = self.model_settings["post_settings"]

    def Run(self):
        self._PrintHeaderMain()

        # Initialize Models to Train
        self._InitializeLinearModel()
        self._InitializeDamageModel()
        self._PrintInfoModel()

        # Import the Input Data
        self._ImportStrainsForPrint()
  
        self.graph = tf.Graph()
        with self.graph.as_default():
            # Build the Placeholders for the Machine Learning model
            self._BuildPlaceholders()

            # Define the models variables
            self._BuildVariableListToPrint()

            # Call the predicted stresses
            self._CallPredictedStresses()

            # Global operations to start the graph
            self._GlobalOpGraphStartToPrint()

        # Initialize the Graph session:
        self._InitializeSessionToPrint()
        
        # Start training loop of linear model
        sig_print = self._RunSessionToPrint()
        return sig_print, self.eps_print

    '''
    ------------------------------------------------------------------------------
    '''

    def _InitializeLinearModel(self):
        le_model = ConLawL.ModelSettings.GetLinearElasticModel(self.model_settings)
        self.le_model_type = ConLawL.ModelSettings.GetLinearElasticType(self.model_settings)
        self.le_model_name = ConLawL.ModelSettings.GetLinearElasticModelName(self.model_settings)

    def _InitializeDamageModel(self):
        damage_model = ConLawL.ModelSettings.GetDamageModel(self.model_settings)

        self.compression_damage_model_type = ConLawL.ModelSettings.GetDamageTypeCompression(self.model_settings)
        self.tension_damage_model_type = ConLawL.ModelSettings.GetDamageTypeTension(self.model_settings)

        self.damage_model_type =  ConLawL.ModelSettings.GetDamageModelType(self.model_settings)
        self.damage_model_name = ConLawL.ModelSettings.GetDamageModelName(self.model_settings)
        
        ''' In case of Bezier Application in damage law, the energy approach of the ML is available'''
        self.bezier_energy_approach = damage_model["bezier_settings"]["comp_energy_approach"]

    def _ImportStrainsForPrint(self):
        Inputs = ConLawL.TrainingInput(self.input_settings)
        if self.input_settings["strain_data"] == "from_boundary":
            self.eps_print = Inputs.GetBoundaryStrains
        elif self.input_settings["strain_data"] == "from_average":
            self.eps_print  = Inputs.GetAverageStrains
    
    def _BuildPlaceholders(self):
        with tf.name_scope("Placeholders"):
            self.EPS = tf.placeholder(tf.float32, name="EPSILON")
            self.SIG = tf.placeholder(tf.float32, name="SIGMA")

    def _BuildVariableListToPrint(self):
        with tf.name_scope("Variables"):
            with tf.name_scope("LinearElasticModelVariables"):
                var_type_le = getattr(ConLawL.ModelVariables(), self.le_model_type)
                self.vars_le = var_type_le(self.init_var_values).Variables
                self.vars_le_limit = var_type_le.ConstrainVariables(self.vars_le, self.init_var_values)

            with tf.name_scope("DamageModelVariables"):
                if self.compression_damage_model_type == "BezierHardSoft" and self.bezier_energy_approach == True:
                    self.var_type_nl = getattr(ConLawL.ModelVariables(), self.damage_model_type + "WithFractureEnergy")
                else: 
                    self.var_type_nl = getattr(ConLawL.ModelVariables(), self.damage_model_type)

                self.vars_nl = self.var_type_nl(self.init_var_values).Variables
                if self.compression_damage_model_type == "BezierHardSoft":
                    self.vars_nl_limit = self.var_type_nl.ConstrainVariables(self.vars_nl, self.vars_le_limit, self.init_var_values)
                else:
                    self.vars_nl_limit = self.var_type_nl.ConstrainVariables(self.vars_nl, self.init_var_values)

    def _CallPredictedStresses(self):
        with tf.name_scope("LinearElasticLaw"):
            le_model = getattr(ConLawL, self.le_model_name)
            with tf.name_scope('PredictedStress'):
                self.SIG_PRED_LE = le_model(self.vars_le_limit).GetStress(self.EPS)
                SIG_EFF = le_model(self.vars_le_limit).GetStress(self.EPS)
        with tf.name_scope("DamageLaw"):    
            nl_model = getattr(ConLawL, self.damage_model_name)
            with tf.name_scope('PredictedStress'):
                self.SIG_PRED_NL = nl_model(self.vars_le_limit, self.vars_nl_limit).GetStress(SIG_EFF)

    def _GlobalOpGraphStartToPrint(self):
        with tf.name_scope("GlobalOps"):
            self.init = tf.global_variables_initializer()

    def _InitializeSessionToPrint(self):
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init)

    def _RunSessionToPrint(self):
        eps = self.eps_print
        sig_print = self.sess.run(self.SIG_PRED_NL, feed_dict= {self.EPS: eps})
        sig_print *= self.input_settings["resize_fac_stress"]

        return sig_print

    def _PrintHeaderMain(self):
        print("------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------")
        print("\n","                       #############################", "\n", \
              "                     ######## ConLawLearn ########", "\n", \
              "                   #############################", "\n")
        print(" MACHINE LEARNING TECHNIQUE TO PREDICT A MACROSCALE CONSTITUTIVE DAMAGE LAW", \
                "\n", "                   FOR A MICROMODELED MASONRY WALL", "\n",\
               "------------------------------------------------------------------------------")
    
    def _PrintInfoModel(self):
        print(" MODEL TO PRINT:")
        print("    Linear Elasticity Theory: ","<<", self.le_model_name, ">>")
        print("    NonLinear Damage Model:   ","<<", self.damage_model_name, ">>", "\n",\
               "------------------------------------------------------------------------------") 
