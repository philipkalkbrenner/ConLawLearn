import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ConLawLearn import ConLawL


class GraphClassic(object):
    def __init__(self, model_settings, initial_variable_values):
        self.model_settings = model_settings
        self.init_var_values = initial_variable_values

    def Run(self):
        print("------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------")
        print("\n","                       #############################", "\n", \
              "                     ######## ConLawLearn ########", "\n", \
              "                   #############################", "\n")
        print(" MACHINE LEARNING TECHNIQUE TO PREDICT A MACROSCALE CONSTITUTIVE DAMAGE LAW", \
                "\n", "                   FOR A MICROMODELED MASONRY WALL", "\n",\
               "------------------------------------------------------------------------------")
        '''
        ------------------------------------------------------------------------------
        Load the Model Settings for the Constitutive Law
        ------------------------------------------------------------------------------
        '''
        input_settings = self.model_settings["input_settings"]

        ''' Linear Model '''
        le_model = ConLawL.ModelSettings.GetLinearElasticModel(self.model_settings)
        le_model_type = ConLawL.ModelSettings.GetLinearElasticType(self.model_settings)
        le_model_name = ConLawL.ModelSettings.GetLinearElasticModelName(self.model_settings)

        ''' Damage/Nonlinear model '''
        damage_model = ConLawL.ModelSettings.GetDamageModel(self.model_settings)
        damage_model_type =  ConLawL.ModelSettings.GetDamageModelType(self.model_settings)
        damage_model_name = ConLawL.ModelSettings.GetDamageModelName(self.model_settings)

        ''' check for bezier application:                                '''
        bezier_applied = damage_model["bezier_settings"]["applied?"]
        bezier_energy_approach = damage_model["bezier_settings"]["comp_energy_approach"]
        bezier_train_controllers = damage_model["bezier_settings"]["train_controllers"]

        ml_settings = self.model_settings["machine_learning_settings"]
        post_settings = self.model_settings["post_settings"]
            
        print(" MODEL TO PREDICT:")
        print("    Linear Elasticity Theory: ","<<", le_model_name, ">>")
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
        eps_le = Inputs.GetStrainsLinearElastic
        sig_le = Inputs.GetStressesLinearElastic
        eps_nl = Inputs.GetStrainsNonlinear
        sig_nl = Inputs.GetStressesNonlinear

        eps_le_train, eps_le_test, sig_le_train, sig_le_test = \
                              Inputs.SplitTrainingAndTesting(eps_le, sig_le)
        eps_nl_train, eps_nl_test, sig_nl_train, sig_nl_test = \
                              Inputs.SplitTrainingAndTesting(eps_nl, sig_nl)

        print(" MODEL TRAINING DATA")
        print(" Total Input Data = ", int(input_settings["train_test_factor"]*100),"%  Training Data + ",  \
              100-int(input_settings["train_test_factor"]*100), "%  Testing Data")
        print("    Linear Elastic:", "\n", \
              "        Training Set Size : ", "<<", eps_le_train.shape[0],  ">>", "States", "\n", \
              "        Testing Set Size  : ", "<<", eps_le_test.shape[0], ">>", "States")
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
        print("    Linear Elastic Parameters:",  "\n",\
              "      --> Feeding Linear Input Strains", "\n",\
              "      --> Predicting Linear Stresses", "\n",\
              "      --> Loss Function with Input Stresses", "\n",\
              "      --> Optimize Linear Parameters to Minimize Loss")
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
            with tf.name_scope("Variables"):
                with tf.name_scope("LinearElasticModelVariables"):
                    var_type_le = getattr(ConLawL.ModelVariables(), le_model_type)
                    vars_le = var_type_le(self.init_var_values).Variables
                    vars_le_plot = var_type_le(self.init_var_values).Vars4Print
                    vars_le_limit = var_type_le.ConstrainVariables(vars_le, self.init_var_values)
            
                with tf.name_scope("DamageModelVariables"):
                    if bezier_energy_approach == "On" and bezier_applied =="Yes":
                        var_type_nl = getattr(ConLawL.ModelVariables(), damage_model_type + "WithFractureEnergy")
                        vars_nl_plot = var_type_nl(self.init_var_values).Vars4Print
                        #print("BezierControlled")
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
                        vars_nl_limit = var_type_nl.ConstrainVariables(vars_nl, vars_le_limit,  self.init_var_values)
                    else:
                        vars_nl_limit = var_type_nl.ConstrainVariables(vars_nl)

            print(" VARIABLES TO OPTIMIZE")
            print(" Total Number of Variables =", len(vars_le_plot) + len(vars_nl_plot))
            print("    Linear Elastic Variables")
            for i in range(len(vars_le_plot)):
                print("       -->", vars_le_plot[i])
            print("    Nonlinear Damage Variables")            
            for i in range(len(vars_nl_plot)):
                print("       -->", vars_nl_plot[i])
            print(" ------------------------------------------------------------------------------")
  
            '''
            --------------------------------------------------------------------------
            Call the predicted stress for the Linear Elastic Law and the Damage Law
            '''
            with tf.name_scope("LinearElasticLaw"):
                le_model = getattr(ConLawL.ConstitutiveLaw(), le_model_name)
                with tf.name_scope('PredictedStress'):
                    SIG_PRED_LE = le_model.GetStress(EPS, vars_le_limit)
                    SIG_EFF = le_model.GetStress(EPS, vars_le_limit)
            with tf.name_scope("DamageLaw"):    
                nl_model = getattr(ConLawL.ConstitutiveLaw(), damage_model_name)
                with tf.name_scope('PredictedStress'):
                    SIG_PRED_NL = nl_model(SIG_EFF, vars_le_limit, vars_nl_limit).GetStress
            '''
            --------------------------------------------------------------------------
            Define the Training and Cost Functions
            '''        
            with tf.name_scope('TrainingFunctions'):
                train_le = tf.reduce_sum(tf.square(tf.subtract(SIG_PRED_LE, \
                                                SIG)), name = "Cost_le")
                train_nl = tf.reduce_sum(tf.square(tf.subtract(SIG_PRED_NL, \
                                                SIG)), name = "Cost_nl")

            '''
            --------------------------------------------------------------------------
            Define the Optimizer to minimize the cost function
            '''
            with tf.name_scope("Optimization"):
                l_rate_le = ml_settings["learn_rate_le"]
                optim_le = getattr(tf.train, ml_settings["optimizer_le"])
                optimizer_le  = optim_le(l_rate_le).minimize(train_le, var_list = vars_le)

                l_rate_nl = ml_settings["learn_rate_nl"]
                optim_nl = getattr(tf.train, ml_settings["optimizer_nl"])
                if bezier_applied == "Yes" and bezier_energy_approach == "On" \
                    and bezier_train_controllers == "No":
                    optimizer_nl  = optim_nl(l_rate_nl).minimize(train_nl, var_list = vars_nl[:-3])
                else:
                    optimizer_nl  = optim_nl(l_rate_nl).minimize(train_nl, var_list = vars_nl)

            print(" GRADIENT OPTIMIZATION")
            print("    Linear Elastic Optimization: ")
            print("        Optimizer:           ", ml_settings["optimizer_le"], '\n',\
                  "       Learning Rate:       ", l_rate_le, '\n',\
                  "       max number of epochs:", ml_settings["max_epoch_le"])
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
        -------------------------------------------------------------------------------
        '''
        print(" ------------------------------------------------------------------------------")
        print(" OPTIMIZATION STARTS")
        print(" ------------------------------------------------------------------------------")
        print("    Linear Elastic Parameters")
        print(" ------------------------------------------------------------------------------")
        
    
        prev_train_cost_le = 0.0
        eps_le_prev = eps_le_train
        sig_le_prev = sess.run(SIG_PRED_LE, feed_dict={EPS:eps_le_train})
        randomizer_le = np.arange(eps_le_train.shape[0])

        for epoch_i in range(n_epochs_le):
            np.random.shuffle(randomizer_le)
            eps_le_rand = eps_le_train[randomizer_le]
            sig_le_rand = sig_le_train[randomizer_le]

            for (inps1, inps2) in zip(eps_le_rand, sig_le_rand):
                eps = [inps1]
                sig = [inps2]
                sess.run(optimizer_le, feed_dict = {EPS:eps, SIG:sig})
            train_cost_le = sess.run(train_le, feed_dict={EPS: eps_le_train, SIG: sig_le_train})
            test_cost_le  = sess.run(train_le, feed_dict={EPS: eps_le_test, SIG: sig_le_test})
            summary = sess.run(merged_summaries, feed_dict={EPS: eps_le_train, SIG: sig_le_train})
            writer.add_summary(summary, global_step = epoch_i)

            sig_le_eval = sess.run(SIG_PRED_LE, feed_dict={EPS:eps_le_prev})

            print("    EPOCH STEP:", epoch_i+1, "\n", \
                  "       Training Cost =", round(train_cost_le/eps_le_train.shape[0], 8), '\n', \
                  "       Testing Cost  =", round(test_cost_le/eps_le_test.shape[0],8))
            print("       Optimized Variable Value:")
            print("          ", sess.run(vars_le_limit))
            print(" ------------------------------------------------------------------------------")
            
            actual_tolerance = np.abs(prev_train_cost_le - train_cost_le)
            if actual_tolerance < ml_settings['learn_crit_le']:
                epoch_le = epoch_i
                break
            prev_train_cost_le = train_cost_le

        print(" OPTIMIZATION OF THE LINEAR ELASTIC VARIABLES FINISHED")
        print("    Final Tolerance of Training Cost: ", actual_tolerance)
        
        '''
            Final Comparison Plot
        '''
        print(eps_le_train[:,1])
        print(sig_le_train[:,1])
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.scatter(eps_le_prev[:,0], sig_le_train[:,0])#, color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax1.scatter(eps_le_prev[:,0], sig_le_eval[:,0])#, color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax1.scatter(eps_le_prev[:,0], sig_le_prev[:,0])#, color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax1.set_title('Plot of Test Data')
        ax1.set_xlabel('stress' r'$\ \epsilon_{xx}\ [-]$')
        ax1.set_ylabel('stress' r'$\ \sigma_{xx}\ [N/mm^2]$')
        ax1.legend(loc = 'lower left')

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(eps_le_prev[:,1], sig_le_train[:,1], color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax2.plot(eps_le_prev[:,1], sig_le_eval[:,1], color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax2.plot(eps_le_prev[:,1], sig_le_prev[:,1], color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax2.set_title('Plot of Test Data')
        ax2.set_xlabel('stress' r'$\ \epsilon_{yy}\ [-]$')
        ax2.set_ylabel('stress' r'$\ \sigma_{yy}\ [N/mm^2]$')
        ax2.legend(loc = 'lower left')

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.plot(eps_le_prev[:,2], sig_le_train[:,2], color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax3.plot(eps_le_prev[:,2], sig_le_eval[:,2], color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax3.plot(eps_le_prev[:,2], sig_le_prev[:,2], color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax3.set_title('Plot of Test Data')
        ax3.set_xlabel('stress' r'$\ \gamma_{xy}\ [-]$')
        ax3.set_ylabel('stress' r'$\ \sigma_{xy}\ [N/mm^2]$')
        ax3.legend(loc = 'lower left')

        plt.show()


        '''
        -------------------------------------------------------------------------------
        Training of the nonlinear damage model choosen in the ModelSettings.json
        -------------------------------------------------------------------------------
        '''


        prev_train_cost_nl = 0.0
        sig_nl_prev = sess.run(SIG_PRED_NL, feed_dict={EPS:eps_nl_train})
        randomizer_nl = np.arange(eps_nl_train.shape[0])        

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

            sig_nl_eval = sess.run(SIG_PRED_NL, feed_dict={EPS:eps_nl_train})


            

            print("EPOCH STEP:", epoch_i, "\n", \
                  "-->",  "training_cost_nl =", train_cost_nl/eps_nl_train.shape[0], '\n', \
                  "-->",  "testing_cost_nl  =", test_cost_nl/eps_nl_test.shape[0], "\n", \
              "Trained Variables:")
            print("    ", sess.run(vars_nl_limit))
            
            if np.abs(prev_train_cost_nl - train_cost_nl) < ml_settings['learn_crit_nl']:
                epoch_nl = epoch_i
                break
            prev_train_cost_nl = train_cost_nl

        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111)
        ax4.scatter(eps_nl_train[:,0], sig_nl_train[:,0],s=1, label = "input model")#, color = 'gray', linewidth = 1, linestyle = '-')
        ax4.scatter(eps_nl_train[:,0], sig_nl_eval[:,0], s=1, label = "optimized model")#, color = 'black',  linewidth = 2, linestyle = '-.')
        ax4.scatter(eps_nl_train[:,0], sig_nl_prev[:,0], s=1, label = "initial model")#, color = 'black',  linewidth = 1, linestyle = '-')
        ax4.set_title('Plot of Test Data')
        ax4.set_xlabel('stress' r'$\ \epsilon_{xx}\ [-]$')
        ax4.set_ylabel('stress' r'$\ \sigma_{xx}\ [N/mm^2]$')
        ax4.legend(loc = 'lower left')

        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111)
        ax5.scatter(eps_nl_train[:,1], sig_nl_train[:,1], s=1, label = "input model")#, color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax5.scatter(eps_nl_train[:,1], sig_nl_eval[:,1], s=1, label = "optimized model")#, color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax5.scatter(eps_nl_train[:,1], sig_nl_prev[:,1], s=1, label = "initial model")#, color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax5.set_title('Plot of Test Data')
        ax5.set_xlabel('stress' r'$\ \epsilon_{yy}\ [-]$')
        ax5.set_ylabel('stress' r'$\ \sigma_{yy}\ [N/mm^2]$')
        ax5.legend(loc = 'lower left')

        fig6 = plt.figure()
        ax6 = fig6.add_subplot(111)
        ax6.scatter(eps_nl_train[:,2], sig_nl_train[:,2], s=1, label = "input model")#, color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax6.scatter(eps_nl_train[:,2], sig_nl_eval[:,2], s=1, label = "optimized model")#, color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax6.scatter(eps_nl_train[:,2], sig_nl_prev[:,2], s=1, label =  "initial model")#, color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax6.set_title('Plot of Test Data')
        ax6.set_xlabel('stress' r'$\ \gamma_{xy}\ [-]$')
        ax6.set_ylabel('stress' r'$\ \sigma_{xy}\ [N/mm^2]$')
        ax6.legend(loc = 'lower left')

        plt.show()
        


    
    
    
    
                           
        
    
        
        
        

    
        
    










