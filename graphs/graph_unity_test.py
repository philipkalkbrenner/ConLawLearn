import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys

import ConLawLearn as ConLawL


class GraphUnityTest(object):
    def __init__(self, unity_settings, unit_test_var_values, model_settings, init_var_values):
        self.unity_settings = unity_settings
        self.unit_test_var_values = unit_test_var_values
        self.model_settings = model_settings
        self.init_var_values = init_var_values

        self.input_settings = self.model_settings["input_settings"]
        self.ml_settings = self.model_settings["machine_learning_settings"]
        self.post_settings = self.model_settings["post_settings"]

    def Run(self):
        self._PrintHeaderMain()

        #Produce Data to Test the Model
        self.epsilon, self.epsilon_test, self.sigma, self.sigma_test = self._ProduceInputData()

        # Initialize Models to Train
        self._InitializeLinearModel()
        self._InitializeDamageModel()
        self._PrintInfoModel()
       
        # Import the Input Data (here already exist -> only called to split in train and test data)
        self._ImportTrainingData()
        self._PrintInfoTrainingData()

        # Construct the Graph
        self._PrintInfoGraphType()

        self.graph = tf.Graph()
        with self.graph.as_default():
            # Build the Placeholders for the Machine Learning model
            self._BuildPlaceholders()

            # Define the models variables
            self._BuildVariableList()
            self._PrintInfoVariableList()

            # Call the predicted stresses
            self._CallPredictedStresses()

            # Define training cost functions
            self._ConstructCostFunction()

            # Define the optimzers
            self._ConstructOptimizer()
            self._PrintInfoOptimizer()

            # Write tensorflow summaries
            self._ConstructSummaryWriter()

            # Global operations to start the graph
            self._GlobalOpGraphStart()
        

        self.log_dir = self.post_settings['tensorboard_logdir']
        self.n_epochs_le = self.ml_settings['linear_elastic_model']['max_epoch']
        self.n_epochs_nl = self.ml_settings['nonlinear_damage_model']['max_epoch']
    
        self._InitializeSession()

        self._PrintInfoStartOptimization()
        
        # Start Training Loop damage model
        self._RunTrainingLoopDamageModel()

        # Plot results
        self._ResultPlot()

    '''
    ------------------------------------------------------------------------------
    '''

    def _InitializeLinearModel(self):
        #self.le_model = ConLawL.ModelSettings.GetLinearElasticModel(self.model_settings)
        self.le_model_type = ConLawL.ModelSettings.GetLinearElasticType(self.model_settings)
        self.le_model_name = ConLawL.ModelSettings.GetLinearElasticModelName(self.model_settings)
        pass

    def _InitializeDamageModel(self):
        damage_model = ConLawL.ModelSettings.GetDamageModel(self.model_settings)
        self.compression_damage_model_type = ConLawL.ModelSettings.GetDamageTypeCompression(self.model_settings)
        self.tension_damage_model_type = ConLawL.ModelSettings.GetDamageTypeTension(self.model_settings)
        self.damage_model_type =  ConLawL.ModelSettings.GetDamageModelType(self.model_settings)
        self.damage_model_name = ConLawL.ModelSettings.GetDamageModelName(self.model_settings)

        # Check for Bezier Application:                                '''
        self.bezier_energy_approach = damage_model["bezier_settings"]["comp_energy_approach"]
        pass
    
    def _ImportTrainingData(self):
        Inputs = ConLawL.TrainingInput(self.input_settings)
        self.eps_nl_train, self.eps_nl_test, self.sig_nl_train, self.sig_nl_test = \
                    Inputs.SplitTrainingAndTesting(self.epsilon, self.sigma)
        pass
    def _BuildPlaceholders(self):
        with tf.name_scope("Placeholders"):
            self.EPS = tf.placeholder(tf.float32, name="EPSILON")
            self.SIG = tf.placeholder(tf.float32, name="SIGMA")
        pass

    def _BuildVariableList(self):
        ''' Linear Elastic Model'''
        var_type_le = getattr(ConLawL.ModelVariables(), self.le_model_type)
        vars_le = var_type_le(self.init_var_values).Variables
        vars_le_plot = var_type_le(self.init_var_values).Vars4Print
        self.vars_le_limit = var_type_le.ConstrainVariables(vars_le, self.init_var_values)
        self.learning_rates_le = var_type_le(self.init_var_values).LearningRates
        self.train_or_not_le = var_type_le(self.init_var_values).TrainRequest

        if len(vars_le) == len(self.learning_rates_le) and len(vars_le) == len(self.train_or_not_le):
            number_of_le_vars = len(vars_le)
        else:
            print("Something is wrong with the Variable list lengths of the linear model, they are not equal!")
            sys.exit()



        for p in range(number_of_le_vars):
            if self.train_or_not_le[p] == False:
                del vars_le[p]
                del self.learning_rates_le[p]
            elif self.train_or_not_le[p] == True:
                pass
            else:
                print("Please define if the variable should be trained or not!", "\n", \
                      "Write <<true>> or <<false>> ")
            

        ''' Nonlinear Damage Model'''
        ''' Build variable list type and list for plot'''

        if self.compression_damage_model_type == "BezierHardSoft" and self.bezier_energy_approach == True:
            self.var_type_nl = getattr(ConLawL.ModelVariables(), self.damage_model_type + "WithFractureEnergy")
        else: 
            self.var_type_nl = getattr(ConLawL.ModelVariables(), self.damage_model_type)
        
        self.vars_nl_plot = self.var_type_nl(self.init_var_values).Vars4Print

        self.vars_nl = self.var_type_nl(self.init_var_values).Variables
        self.learning_rates_nl = self.var_type_nl(self.init_var_values).LearningRates
        self.train_or_not_nl = self.var_type_nl(self.init_var_values).TrainRequest

        
    
        if self.compression_damage_model_type == "BezierHardSoft":
            self.vars_nl_limit = self.var_type_nl.ConstrainVariables(self.vars_nl, self.vars_le_limit, self.init_var_values)
        else:
            self.vars_nl_limit = self.var_type_nl.ConstrainVariables(self.vars_nl, self.init_var_values)
        pass

        if len(self.vars_nl) == len(self.learning_rates_nl) and len(self.vars_nl) == len(self.train_or_not_nl):
            number_of_nl_vars = len(self.vars_nl)
        else:
            print("Something is wrong with the Variable list lengths of the nonlinear model, they are not equal!")
            sys.exit()

        self.vars_nl = [ \
            b for a, b in zip(self.train_or_not_nl, self.vars_nl) if a \
            ]
        self.learning_rates_nl = [ \
            b for a, b in zip(self.train_or_not_nl, self.learning_rates_nl) if a \
            ]

    def _CallPredictedStresses(self):
        le_model = getattr(ConLawL, self.le_model_name)
        self.SIG_EFF = le_model(self.vars_le_limit).GetStress(self.EPS)
   
        nl_model = getattr(ConLawL, self.damage_model_name)
        self.SIG_PRED_NL = nl_model(self.vars_le_limit, self.vars_nl_limit).GetStress(self.SIG_EFF)
        pass
    
    def _ConstructCostFunction(self):
        train_nl_error = tf.subtract(self.SIG_PRED_NL, self.SIG)
        train_nl_error_abs = tf.abs(train_nl_error)
        train_nl_square = tf.square(train_nl_error_abs)

        self.train_nl = tf.reduce_sum(train_nl_square)
        pass
    
    def _ConstructOptimizer(self):
        '''
        Construction of the optimizer:
        A) Train all Variables at same step with same learning rate 
           (one step each optimization , is batched)
        B) Train each Variable separately with different training rates 
           (n steps each optimization, is batched, n = number of variables)
        '''
        
        self.l_rate_nl = self.ml_settings['nonlinear_damage_model']['learning_rate']
        optim_nl = getattr(tf.train, self.ml_settings['nonlinear_damage_model']['optimizer'])
        self.optimizer_nl = []

        if self.ml_settings['seperated_variable_training'] == False:
            self.optimizer_nl.append(optim_nl(self.l_rate_nl).minimize(self.train_nl, var_list = self.vars_nl))
            pass
        elif self.ml_settings['seperated_variable_training'] == True:
            self.learning_rates_nl = [i * self.l_rate_nl for i in self.learning_rates_nl]
            for i in range(len(self.vars_nl)):
                optimizer_i = optim_nl(self.learning_rates_nl[i]). minimize(self.train_nl, var_list = self.vars_nl[i])
                self.optimizer_nl.append(optimizer_i)
            pass
        else:
            print("Error: Please define if you want to train the nonlinear damage variables in seperate steps or not!", "\n", \
            "By indicating it with <<Yes>> or <<No>> in the train_seperately_nl? field.")
            sys.exit()
    
    def _ConstructSummaryWriter(self):
        with tf.name_scope('AllSummaries'):
            self.sum_writer_le = getattr(ConLawL.ModelVariables(), self.le_model_type + "Summary")
            self.sum_writer_le(self.vars_le_limit)

            if self.compression_damage_model_type == "BezierHardSoft" and self.bezier_energy_approach == True:
                self.sum_writer_nl = getattr(ConLawL.ModelVariables(), self.damage_model_type + \
                                    "WithFractureEnergy" + "Summary")
            else:
                self.sum_writer_nl = getattr(ConLawL.ModelVariables(), self.damage_model_type + \
                                "Summary")
            self.sum_writer_nl(self.vars_nl_limit)
        pass
        

    def _GlobalOpGraphStart(self):
        with tf.name_scope("GlobalOps"):
            self.init = tf.global_variables_initializer()
            self.merged_summaries = tf.summary.merge_all()

    def _InitializeSession(self):
        self.sess = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter(self.log_dir, self.graph)

        self.sess.run(self.init)
    
    def _RunTrainingLoopDamageModel(self):
        prev_train_cost_nl = 0.0
        randomizer_nl = np.arange(self.eps_nl_train.shape[0])
        self.sigma_prev = self.sess.run(self.SIG_PRED_NL, feed_dict={self.EPS:self.epsilon_test})
        print(self.sess.run(self.train_nl, feed_dict={self.EPS:self.eps_nl_train, self.SIG:self.sig_nl_train}))

        for epoch_i in range(self.n_epochs_nl):
            np.random.shuffle(randomizer_nl)
            eps_nl_rand = self.eps_nl_train[randomizer_nl]
            sig_nl_rand = self.sig_nl_train[randomizer_nl]
            for (inps1, inps2) in zip(eps_nl_rand, sig_nl_rand):
                eps = [inps1]
                sig = [inps2]
                for optimizer_i in range(len(self.optimizer_nl)):
                    self.sess.run(self.optimizer_nl[optimizer_i], feed_dict = {self.EPS:eps, self.SIG:sig})
        
            train_cost_nl = self.sess.run(self.train_nl, feed_dict={self.EPS: self.eps_nl_train, self.SIG: self.sig_nl_train})
            test_cost_nl  = self.sess.run(self.train_nl, feed_dict={self.EPS: self.eps_nl_test, self.SIG: self.sig_nl_test})
            summary = self.sess.run(self.merged_summaries, feed_dict={self.EPS: self.eps_nl_train, self.SIG: self.sig_nl_train})
            self.writer.add_summary(summary, global_step = epoch_i)

            self.sigma_eval = self.sess.run(self.SIG_PRED_NL, feed_dict={self.EPS:self.epsilon_test})

            print("EPOCH STEP:", epoch_i, "\n", \
                  "-->",  "training_cost_nl =", train_cost_nl/self.eps_nl_train.shape[0], '\n', \
                  "-->",  "testing_cost_nl  =", test_cost_nl/self.eps_nl_test.shape[0], "\n", \
                    "Trained Variables:")
            print("      ",self.sess.run(self.vars_nl_limit))
            
            if np.abs(prev_train_cost_nl - train_cost_nl) < self.ml_settings['nonlinear_damage_model']['learn_crit']:
                epoch_nl = epoch_i
                break
            prev_train_cost_nl = train_cost_nl

    def _ResultPlot(self):
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(self.epsilon_test[:,0], self.sigma_test[:,0], color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax1.plot(self.epsilon_test[:,0], self.sigma_eval[:,0], color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax1.plot(self.epsilon_test[:,0], self.sigma_prev[:,0], color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax1.set_title('Plot of Test Data')
        ax1.set_xlabel('stress' r'$\ \epsilon_{xx}\ [-]$')
        ax1.set_ylabel('stress' r'$\ \sigma_{xx}\ [N/mm^2]$')
        ax1.legend(loc = 'lower left')

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(self.epsilon_test[:,1], self.sigma_test[:,1], color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax2.plot(self.epsilon_test[:,1], self.sigma_eval[:,1], color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax2.plot(self.epsilon_test[:,1], self.sigma_prev[:,1], color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax2.set_title('Plot of Test Data')
        ax2.set_xlabel('stress' r'$\ \epsilon_{yy}\ [-]$')
        ax2.set_ylabel('stress' r'$\ \sigma_{yy}\ [N/mm^2]$')
        ax2.legend(loc = 'lower left')

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.plot(self.epsilon_test[:,2], self.sigma_test[:,2], color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax3.plot(self.epsilon_test[:,2], self.sigma_eval[:,2], color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax3.plot(self.epsilon_test[:,2], self.sigma_prev[:,2], color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax3.set_title('Plot of Test Data')
        ax3.set_xlabel('stress' r'$\ \gamma_{xy}\ [-]$')
        ax3.set_ylabel('stress' r'$\ \sigma_{xy}\ [N/mm^2]$')
        ax3.legend(loc = 'lower left')

        plt.show()


    def _ProduceInputData(self):
        le_model_type = ConLawL.ModelSettings.GetLinearElasticType(self.unity_settings)
        le_model_name = ConLawL.ModelSettings.GetLinearElasticModelName(self.unity_settings)

        damage_model = ConLawL.ModelSettings.GetDamageModel(self.unity_settings)
        damage_model_type =  ConLawL.ModelSettings.GetDamageModelType(self.unity_settings)
        damage_model_name = ConLawL.ModelSettings.GetDamageModelName(self.unity_settings)

        bezier_applied = damage_model["bezier_settings"]["applied?"]
        bezier_energy_approach = damage_model["bezier_settings"]["comp_energy_approach"]
        bezier_train_controllers = damage_model["bezier_settings"]["train_controllers"]

        unit_test_settings = self.unity_settings["input_data_producuction"]

        epsilon_test = ConLawL.RandomStrainGenerator.GetRandomStrainForPlot(0.09)
        # Strains as Input for the Model to train
        epsilon = ConLawL.RandomStrainGenerator(unit_test_settings).GetStrain

        # Tensorflow Placeholder of Data Production
        EPS = tf.placeholder(tf.float32, name="EPSILON")

        # Construct the Variable Lists
        var_type_le = getattr(ConLawL.ModelVariables(), le_model_type)
        vars_le = var_type_le(self.unit_test_var_values).Variables
        #vars_le_plot = var_type_le(self.unit_test_var_values).Vars4Print
        vars_le_limit = var_type_le.ConstrainVariables(vars_le, self.unit_test_var_values)

        if bezier_energy_approach =="On" and bezier_applied =="Yes":
            var_type_nl = getattr(ConLawL.ModelVariables(), damage_model_type + "WithFractureEnergy")
        elif bezier_energy_approach =="Off" and bezier_applied=="Yes":
            var_type_nl = getattr(ConLawL.ModelVariables(), damage_model_type)
        else:
            if bezier_applied == "Yes":
                print(" WARNING: Error in ModelSettings.Json !!!", "\n",\
                    "Please define the comp_energy_approach in ModelSettings.json as On or Off!")
                sys.exit()
            else:
                var_type_nl = getattr(ConLawL.ModelVariables(), damage_model_type)
        
        vars_nl = var_type_nl(self.unit_test_var_values).Variables

        if bezier_applied == "Yes":
            vars_nl_limit = var_type_nl.ConstrainVariables(vars_nl, vars_le_limit, self.unit_test_var_values)
        else:
            vars_nl_limit = var_type_nl.ConstrainVariables(vars_nl, self.unit_test_var_values)


        # Call the model linear and nonlinear model classes
        le_model = getattr(ConLawL, le_model_name)
        nl_model = getattr(ConLawL, damage_model_name)

        # Compute the Stresses
        SIG_EFF = le_model(vars_le_limit).GetStress(EPS)
        SIG_PRED_NL = nl_model(vars_le_limit, vars_nl_limit).GetStress(SIG_EFF)

        # Initialize the Tensorflow Session
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        # Fill the stresses
        sigma = sess.run(SIG_PRED_NL, feed_dict={EPS:epsilon})
        sigma_test = sess.run(SIG_PRED_NL, feed_dict={EPS:epsilon_test})

        # Putting Noise to the stress values
        noise_factor = unit_test_settings["noise_factor"]
        noise = np.random.randint(-1.0, 2.0, (sigma.shape[0], sigma.shape[1]))*noise_factor
        sigma = sigma + noise

        return (epsilon, epsilon_test, sigma, sigma_test)

    def _PrintHeaderMain(self):
        print("------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------")
        print("\n","                       #############################", "\n", \
              "                     ######## ConLawLearn ########", "\n", \
              "                   #############################", "\n")
        print(" MACHINE LEARNING TECHNIQUE TO PREDICT A MACROSCALE CONSTITUTIVE DAMAGE LAW", \
                "\n", "                   FOR A MICROMODELED MASONRY WALL", "\n",\
                "\n", "                               UNIT TEST"          , "\n",\
               "------------------------------------------------------------------------------")

    def _PrintInfoModel(self):
        print(" MODEL TO PREDICT:")
        print("    NonLinear Damage Model:   ","<<", self.damage_model_name, ">>", "\n",\
               "------------------------------------------------------------------------------")    

    def _PrintInfoTrainingData(self):
        print(" MODEL TRAINING DATA")
        print(" Total Input Data = ", int(self.input_settings["train_test_factor"]*100),"%  Training Data + ",  \
                    100-int(self.input_settings["train_test_factor"]*100), "%  Testing Data")
        print("    Nonlinear:", "\n", \
              "        Training Set Size : ", "<<", self.eps_nl_train.shape[0],">>", "States" , "\n", \
              "        Testing Set Size  : ", "<<", self.eps_nl_test.shape[0],">>",  "States")
        print(" ------------------------------------------------------------------------------")

    def _PrintInfoGraphType(self):
        print(" GRAPH CONSTRUCTION TYPE")
        '''
        print("    Linear Elastic Parameters:",  "\n",\
              "      --> Feeding Linear Input Strains", "\n",\
              "      --> Predicting Linear Stresses", "\n",\
              "      --> Loss Function with Input Stresses", "\n",\
              "      --> Optimize Linear Parameters to Minimize Loss")
              '''
        print("    Nonlinear Damage Parameters:", "\n",\
              "      --> Feeding Input Strains", "\n",\
              "      --> Predicting Damage Stress", "\n",\
              "      --> Loss Function with Input Stresses", "\n",\
              "      --> Optimize Damage Parameters to Minimize Loss")
        print(" ------------------------------------------------------------------------------")

    def _PrintInfoVariableList(self):
        print(" VARIABLES TO OPTIMIZE")
        print(" Total Number of Variables =", len(self.vars_nl_plot))
        print("    Nonlinear Damage Variables")            
        for i in range(len(self.vars_nl_plot)):
            print("       -->", self.vars_nl_plot[i])
        print(" ------------------------------------------------------------------------------")

    def _PrintInfoOptimizer(self):
        print(" GRADIENT OPTIMIZATION")
        print("    Nonlinear Optimization: ")
        print("        Optimizer:           ", self.ml_settings['nonlinear_damage_model']['optimizer'], '\n',\
              "       Learning Rate:       ", self.l_rate_nl, '\n',\
              "       max number of epochs:", self.ml_settings['nonlinear_damage_model']['max_epoch'])
        print(" ------------------------------------------------------------------------------")

    def _PrintInfoStartOptimization(self):
        print(" ------------------------------------------------------------------------------")
        print(" OPTIMIZATION STARTS")
        print(" ------------------------------------------------------------------------------")
        '''
        print("    Linear Elastic Parameters")
        print(" ------------------------------------------------------------------------------")
        '''
        