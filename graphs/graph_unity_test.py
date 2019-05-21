import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ConLawLearn import ConLawL
import sys


class GraphUnityTest(object):
    def __init__(self, unity_settings, unit_test_var_values, model_settings, init_var_values):
        self.unity_settings = unity_settings
        self.unit_test_var_values = unit_test_var_values
        self.model_settings = model_settings
        self.init_var_values = init_var_values

    def Run(self):
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
        '''
        ------------------------------------------------------------------------------
        STEP 1: Produce Data to Test the Model
        ------------------------------------------------------------------------------
        '''
        # Construct Model
        
        le_model = ConLawL.ModelSettings.GetLinearElasticModel(self.unity_settings)
        le_model_type = ConLawL.ModelSettings.GetLinearElasticType(self.unity_settings)
        le_model_name = ConLawL.ModelSettings.GetLinearElasticModelName(self.unity_settings)

        damage_model = ConLawL.ModelSettings.GetDamageModel(self.unity_settings)
        damage_model_type =  ConLawL.ModelSettings.GetDamageModelType(self.unity_settings)
        damage_model_name = ConLawL.ModelSettings.GetDamageModelName(self.unity_settings)

        bezier_applied = damage_model["bezier_settings"]["applied?"]
        bezier_energy_approach = damage_model["bezier_settings"]["comp_energy_approach"]
        bezier_train_controllers = damage_model["bezier_settings"]["train_controllers"]

        # Produce Data
        unit_test_settings = self.unity_settings["input_data_producuction"]
        # Strains as test values after optimization
        epsilon_test = ConLawL.RandomStrainGenerator.GetRandomStrainForPlot()
        ###epsilon_test1 = ConLawL.RandomStrainGenerator.GetPureCompressionStrain()
        # Strains as Input for the Model to train
        epsilon = ConLawL.RandomStrainGenerator(unit_test_settings).GetStrain

        # Tensorflow Placeholder of Data Production
        EPS = tf.placeholder(tf.float32, name="EPSILON")

        # Construct the Variable Lists
        var_type_le = getattr(ConLawL.ModelVariables(), le_model_type)
        vars_le = var_type_le(self.unit_test_var_values).Variables
        vars_le_plot = var_type_le(self.unit_test_var_values).Vars4Print
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
        le_model = getattr(ConLawL.ConstitutiveLaw(), le_model_name)
        nl_model = getattr(ConLawL.ConstitutiveLaw(), damage_model_name)

        # Compute the Stresses
        SIG_EFF = le_model.GetStress(EPS, vars_le_limit)
        SIG_PRED_NL = nl_model(SIG_EFF, vars_le_limit, vars_nl_limit).GetStress

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

        '''
        ------------------------------------------------------------------------------
        STEP 1 FINISHED
        ------------------------------------------------------------------------------
        '''
        # ----------------------------------------------------------------------------
        '''
        ------------------------------------------------------------------------------
        STEP 2: Use input Data to Train model
        ------------------------------------------------------------------------------
        '''

        input_settings = self.model_settings["input_settings"]

        # Linear Model
        le_model = ConLawL.ModelSettings.GetLinearElasticModel(self.model_settings)
        le_model_type = ConLawL.ModelSettings.GetLinearElasticType(self.model_settings)
        le_model_name = ConLawL.ModelSettings.GetLinearElasticModelName(self.model_settings)

        # Damage/Nonlinear Model
        damage_model = ConLawL.ModelSettings.GetDamageModel(self.model_settings)
        damage_model_type =  ConLawL.ModelSettings.GetDamageModelType(self.model_settings)
        damage_model_name = ConLawL.ModelSettings.GetDamageModelName(self.model_settings)

        # Check for Bezier Application:                                '''
        bezier_applied = damage_model["bezier_settings"]["applied?"]
        bezier_energy_approach = damage_model["bezier_settings"]["comp_energy_approach"]
        bezier_train_controllers = damage_model["bezier_settings"]["train_controllers"]

        ml_settings = self.model_settings["machine_learning_settings"]
        post_settings = self.model_settings["post_settings"]
    
        print(" MODEL TO PREDICT:")
        print("    NonLinear Damage Model:   ","<<", damage_model_name, ">>", "\n",\
               "------------------------------------------------------------------------------")    
       
    
        '''
        ------------------------------------------------------------------------------
        Import the Input Data
        ------------------------------------------------------------------------------
        Resizing the Stress Input, from Kratos stress arrives in N/m^2 (= Pa).
        Finally a value in MN/m^2  ----->  N/m^2 * 1e-6 = 1.0 MN/m^2 (= MPa)
        see also resize_fac_stress in ModelSettings.json
        '''
        Inputs = ConLawL.TrainingInput(input_settings)
        eps_nl_train, eps_nl_test, sig_nl_train, sig_nl_test = \
                    Inputs.SplitTrainingAndTesting(epsilon, sigma)

        print(" MODEL TRAINING DATA")
        print(" Total Input Data = ", int(input_settings["train_test_factor"]*100),"%  Training Data + ",  \
                    100-int(input_settings["train_test_factor"]*100), "%  Testing Data")
        print("    Nonlinear:", "\n", \
              "        Training Set Size : ", "<<", eps_nl_train.shape[0],">>", "States" , "\n", \
              "        Testing Set Size  : ", "<<", eps_nl_test.shape[0],">>",  "States")
        print(" ------------------------------------------------------------------------------")
        
        '''
        ------------------------------------------------------------------------------
        Construct the Graph
        ------------------------------------------------------------------------------
        '''
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
        graph = tf.Graph()
        with graph.as_default():
            '''
            --------------------------------------------------------------------------
            Build the Placeholders for the Machine Learning model
            '''
            with tf.name_scope("Placeholders"):
                EPS = tf.placeholder(tf.float32, name="EPSILON")
                SIG = tf.placeholder(tf.float32, name="SIGMA")
            '''
            --------------------------------------------------------------------------
            Define the models variables
            '''
    
            var_type_le = getattr(ConLawL.ModelVariables(), le_model_type)
            vars_le = var_type_le(self.init_var_values).Variables
            vars_le_plot = var_type_le(self.init_var_values).Vars4Print
            vars_le_limit = var_type_le.ConstrainVariables(vars_le, self.init_var_values)

            if bezier_energy_approach =="On" and bezier_applied =="Yes":
                var_type_nl = getattr(ConLawL.ModelVariables(), damage_model_type + "WithFractureEnergy")
                vars_nl_plot = var_type_nl(self.init_var_values).Vars4Print
                if bezier_train_controllers =="No":
                    vars_nl_plot = vars_nl_plot[:-3]
            
            elif bezier_energy_approach =="Off" and bezier_applied=="Yes":
                var_type_nl = getattr(ConLawL.ModelVariables(), damage_model_type)
                vars_nl_plot = var_type_nl(self.init_var_values).Vars4Print
            else:
                if bezier_applied == "Yes":
                    print(" WARNING: Error in ModelSettings.Json !!!", "\n",\
                    "Please define the comp_energy_approach in ModelSettings.json as On or Off!")
                    sys.exit()
                else:
                    var_type_nl = getattr(ConLawL.ModelVariables(), damage_model_type)
                    vars_nl_plot = var_type_nl(self.init_var_values).Vars4Print
        
            vars_nl = var_type_nl(self.init_var_values).Variables
    
    
            if bezier_applied == "Yes":
                vars_nl_limit = var_type_nl.ConstrainVariables(vars_nl, vars_le_limit, self.init_var_values)
            else:
                vars_nl_limit = var_type_nl.ConstrainVariables(vars_nl, self.init_var_values)
    

            print(" VARIABLES TO OPTIMIZE")
            print(" Total Number of Variables =", len(vars_nl_plot))
            print("    Nonlinear Damage Variables")            
            for i in range(len(vars_nl_plot)):
                print("       -->", vars_nl_plot[i])
            print(" ------------------------------------------------------------------------------")
    

            '''
            --------------------------------------------------------------------------
            Call the predicted stress for the Linear Elastic Law and the Damage Law
            '''
    
            le_model = getattr(ConLawL.ConstitutiveLaw(), le_model_name)
    
            #SIG_PRED_LE = le_model.GetStress(EPS, vars_le_limit)
            SIG_EFF = le_model.GetStress(EPS, vars_le_limit)
   
            nl_model = getattr(ConLawL.ConstitutiveLaw(), damage_model_name)
        
            SIG_PRED_NL = nl_model(SIG_EFF, vars_le_limit, vars_nl_limit).GetStress

            '''
            --------------------------------------------------------------------------
            Define the Training and Cost Functions
            '''        
            train_nl_error = tf.subtract(SIG_PRED_NL, SIG)
            train_nl_error_abs = tf.abs(train_nl_error)
            train_nl_square = tf.square(train_nl_error_abs)

            train_nl = tf.reduce_sum(train_nl_square)

            '''
            --------------------------------------------------------------------------
            Define the Optimizer to minimize the cost function
            '''

            l_rate_nl = ml_settings["learn_rate_nl"]
            optim_nl = getattr(tf.train, ml_settings["optimizer_nl"])
            if bezier_applied == "Yes" and bezier_energy_approach == "On" \
                and bezier_train_controllers == "No":
                optimizer_nl  = optim_nl(l_rate_nl).minimize(train_nl, var_list = vars_nl[:-3])
            else:
                optimizer_nl  = optim_nl(l_rate_nl).minimize(train_nl, var_list = vars_nl)
    
    
            print(" GRADIENT OPTIMIZATION")
            print("    Nonlinear Optimization: ")
            print("        Optimizer:           ", ml_settings["optimizer_nl"], '\n',\
                  "       Learning Rate:       ", l_rate_nl, '\n',\
                  "       max number of epochs:", ml_settings["max_epoch_nl"])
            print(" ------------------------------------------------------------------------------")
    
            '''
            --------------------------------------------------------------------------
            Write Variable summaries during optimization
            '''
            with tf.name_scope('AllSummaries'):
                sum_writer_le = getattr(ConLawL.ModelVariables(), le_model_type + "Summary")
                sum_writer_le(vars_le_limit)
                if bezier_applied == "Yes" and bezier_energy_approach == "On":
                    sum_writer_nl = getattr(ConLawL.ModelVariables(), damage_model_type + \
                                        "WithFractureEnergy" + "Summary")
                else:
                    sum_writer_nl = getattr(ConLawL.ModelVariables(), damage_model_type + \
                                    "Summary")
                sum_writer_nl(vars_nl_limit)
            '''
            ---------------------------------------------------------------------------
            Global Operation to start the Graph
            '''
            with tf.name_scope("GlobalOps"):
                init = tf.global_variables_initializer()
                merged_summaries = tf.summary.merge_all()


        '''
        -------------------------------------------------------------------------------
        -------------------------------------------------------------------------------
        The GraphRunner starts from here
        -------------------------------------------------------------------------------
        '''

        log_dir = post_settings["tensorboard_logdir"]
        n_epochs_le = ml_settings["max_epoch_le"]
        n_epochs_nl = ml_settings["max_epoch_nl"]
    
        sess = tf.Session(graph=graph)
        writer = tf.summary.FileWriter(log_dir, graph)

        sess.run(init)

        '''
        -------------------------------------------------------------------------------
        Linear Elastic Parameters are trained in a previous step
        by only considering the input data at load steps in the linear range
        ------------------------------------------------------------------------------
        '''
        print(" ------------------------------------------------------------------------------")
        print(" OPTIMIZATION STARTS")
        print(" ------------------------------------------------------------------------------")
        '''
        print("    Linear Elastic Parameters")
        print(" ------------------------------------------------------------------------------")
        ''' 
        

        '''
        -------------------------------------------------------------------------------
        Training of the nonlinear damage model choosen in the ModelSettings.json
        -------------------------------------------------------------------------------
        '''

        prev_train_cost_nl = 0.0
        randomizer_nl = np.arange(eps_nl_train.shape[0])
        sigma_prev = sess.run(SIG_PRED_NL, feed_dict={EPS:epsilon_test})
        print(sess.run(train_nl, feed_dict={EPS:eps_nl_train, SIG:sig_nl_train}))

        for epoch_i in range(n_epochs_nl):
            np.random.shuffle(randomizer_nl)
            eps_nl_rand = eps_nl_train[randomizer_nl]
            sig_nl_rand = sig_nl_train[randomizer_nl]     
            for (inps1, inps2) in zip(eps_nl_rand, sig_nl_rand):
                eps = [inps1]
                sig = [inps2]
                sess.run(optimizer_nl, feed_dict = {EPS:eps, SIG:sig})
        
            train_cost_nl = sess.run(train_nl, feed_dict={EPS: eps_nl_train, SIG: sig_nl_train})
            test_cost_nl  = sess.run(train_nl, feed_dict={EPS: eps_nl_test, SIG: sig_nl_test})
            summary = sess.run(merged_summaries, feed_dict={EPS: eps_nl_train, SIG: sig_nl_train})
            writer.add_summary(summary, global_step = epoch_i)

            sigma_eval = sess.run(SIG_PRED_NL, feed_dict={EPS:epsilon_test})

            print("EPOCH STEP:", epoch_i, "\n", \
                  "-->",  "training_cost_nl =", train_cost_nl/eps_nl_train.shape[0], '\n', \
                  "-->",  "testing_cost_nl  =", test_cost_nl/eps_nl_test.shape[0], "\n", \
                    "Trained Variables:")
            print("      ",sess.run(vars_nl_limit))
            
            if np.abs(prev_train_cost_nl - train_cost_nl) < ml_settings['learn_crit_nl']:
                epoch_nl = epoch_i
                break
            prev_train_cost_nl = train_cost_nl
    

        '''
            Final Comparison Plot
        '''

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.plot(epsilon_test[:,0], sigma_test[:,0], color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax1.plot(epsilon_test[:,0], sigma_eval[:,0], color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax1.plot(epsilon_test[:,0], sigma_prev[:,0], color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax1.set_title('Plot of Test Data')
        ax1.set_xlabel('stress' r'$\ \epsilon_{xx}\ [-]$')
        ax1.set_ylabel('stress' r'$\ \sigma_{xx}\ [N/mm^2]$')
        ax1.legend(loc = 'lower left')

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(epsilon_test[:,1], sigma_test[:,1], color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax2.plot(epsilon_test[:,1], sigma_eval[:,1], color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax2.plot(epsilon_test[:,1], sigma_prev[:,1], color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax2.set_title('Plot of Test Data')
        ax2.set_xlabel('stress' r'$\ \epsilon_{yy}\ [-]$')
        ax2.set_ylabel('stress' r'$\ \sigma_{yy}\ [N/mm^2]$')
        ax2.legend(loc = 'lower left')

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.plot(epsilon_test[:,2], sigma_test[:,2], color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax3.plot(epsilon_test[:,2], sigma_eval[:,2], color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax3.plot(epsilon_test[:,2], sigma_prev[:,2], color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax3.set_title('Plot of Test Data')
        ax3.set_xlabel('stress' r'$\ \gamma_{xy}\ [-]$')
        ax3.set_ylabel('stress' r'$\ \sigma_{xy}\ [N/mm^2]$')
        ax3.legend(loc = 'lower left')

        plt.show()
