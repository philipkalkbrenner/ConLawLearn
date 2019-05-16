import tensorflow as tf
import numpy as np
import json
import ConLawL


class GraphSplitApproach(object):
    def Run():
        print("------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------")
        print("------------------------------------------------------------------------------")
        print("\n","                       #############################", "\n", \
              "                     ######## ConLawLearn ########", "\n", \
              "                   #############################", "\n")
        print(" MACHINE LEARNING TECHNIQUE TO PREDICT A MACROSCALE CONSTITUTIVE DAMAGE LAW", \
                "\n", "FOR A MICROMODELED MASONRY WALL", "\n",\
               "------------------------------------------------------------------------------")
        '''
        ------------------------------------------------------------------------------
        Load the Model Settings for the Constitutive Law
        ------------------------------------------------------------------------------
        '''
        with open('ModelSettings.json') as m:
            model_settings = json.load(m)
    
        input_settings = model_settings["input_settings"]

        ''' Linear Model '''
        le_model = ConLawL.ModelSettings.GetLinearElasticModel(model_settings)
        le_model_type = ConLawL.ModelSettings.GetLinearElasticType(model_settings)
        le_model_name = ConLawL.ModelSettings.GetLinearElasticModelName(model_settings)

        ''' Damage/Nonlinear model '''
        damage_model = ConLawL.ModelSettings.GetDamageModel(model_settings)
        damage_model_type =  ConLawL.ModelSettings.GetDamageModelType(model_settings)
        damage_model_name = ConLawL.ModelSettings.GetDamageModelName(model_settings)

        ''' check for bezier application:                                '''
        bezier_applied = damage_model["bezier_settings"]["applied?"]
        bezier_controlled = damage_model["bezier_settings"]["controlled"]
        bezier_train_controllers = damage_model["bezier_settings"]["train_controllers"]

        ml_settings = model_settings["machine_learning_settings"]
        post_settings = model_settings["post_settings"]

        with open('InitialVariableValues.json') as f:
            init_var_values = json.load(f)

            
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
        

        Inputs      = ConLawL.TrainingInput(input_settings)

        eps_le      = Inputs.GetStrainsLinearElastic
        sig_le      = Inputs.GetStressesLinearElastic
        sig_pos_le  = Inputs.GetPositeStressLinear
        sig_neg_le  = Inputs.GetNegativeStressLinear
        eps_nl      = Inputs.GetStrainsNonlinear
        sig_nl      = Inputs.GetStressesNonlinear
        sig_pos_nl  = Inputs.GetPositeStressNonlinear
        sig_neg_nl  = Inputs.GetNegativeStressNonlinear


        eps_le_train, eps_le_test, sig_le_train, sig_le_test = \
            Inputs.SplitTrainingAndTesting(eps_le, sig_le)
        eps_nl_train, eps_nl_test, \
        sig_pos_nl_train, sig_pos_nl_test, \
        sig_neg_nl_train, sig_neg_nl_test = \
            Inputs.SplitTrainingAndTesting3Entries(eps_nl, sig_pos_nl, sig_neg_nl)
        sig_nl_train = sig_pos_nl_train + sig_neg_nl_train
        sig_nl_test  = sig_pos_nl_test + sig_neg_nl_test

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
                EPS         = tf.placeholder(tf.float32, name="EPSILON")
                SIG_POS = tf.placeholder(tf.float32, name = 'SIGMA_POS')
                SIG_NEG = tf.placeholder(tf.float32, name = 'SIGMA_NEG')
                SIG         = tf.placeholder(tf.float32, name="SIGMA")
            '''
            --------------------------------------------------------------------------
            Define the models variables
            '''
            with tf.name_scope("Variables"):
                with tf.name_scope("LinearElasticModelVariables"):
                    var_type_le = getattr(ConLawL.ModelVariables(), le_model_type)
                    vars_le = var_type_le(init_var_values).Variables
                    vars_le_plot = var_type_le(init_var_values).Vars4Print
                    vars_le_limit = var_type_le.ConstrainVariables(vars_le, init_var_values)
            
                with tf.name_scope("DamageModelVariables"):
                    if bezier_controlled == "Yes":
                        var_type_nl = getattr(ConLawL.ModelVariables(), damage_model_type + "Controlled")
                        print("BezierControlled")
                    else:
                        var_type_nl = getattr(ConLawL.ModelVariables(), damage_model_type)
                
                    vars_nl = var_type_nl(init_var_values).Variables
                    vars_nl_plot = var_type_nl(init_var_values).Vars4Print
            
                    if bezier_applied == "Yes":
                        vars_nl_limit = var_type_nl.ConstrainVariables(vars_nl, vars_le_limit)
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
                    SIG_PRED_POS_NL = nl_model(SIG_EFF, vars_le_limit, vars_nl_limit).GetPositiveStress
                    SIG_PRED_NEG_NL = nl_model(SIG_EFF, vars_le_limit, vars_nl_limit).GetNegativeStress
                    TAU_POS = nl_model(SIG_EFF, vars_le_limit, vars_nl_limit).GetPositiveEquivalentStress
                    TAU_NEG = nl_model(SIG_EFF, vars_le_limit, vars_nl_limit).GetNegativeEquivalentStress  
                    D_POS = nl_model(SIG_EFF, vars_le_limit, vars_nl_limit).GetPositiveDamage
                    D_NEG = nl_model(SIG_EFF, vars_le_limit, vars_nl_limit).GetNegativeDamage
                    S_1,S_2 = ConLawL.YieldCriterion._s1_s2(SIG_EFF)
                    S_Max = ConLawL.YieldCriterion._Smax(S_1, S_2)
                    
            '''
            --------------------------------------------------------------------------
            Define the Training and Cost Functions
            '''        
            with tf.name_scope('TrainingFunctions'):
                train_le = tf.reduce_sum(tf.square(tf.subtract(SIG_PRED_LE, \
                                                SIG)), name = "Cost_le")
                train_nl_pos = tf.reduce_sum(tf.square(tf.subtract(SIG_PRED_POS_NL, \
                                                SIG_POS)), name = "Cost_nl_pos")
                train_nl_neg = tf.reduce_sum(tf.square(tf.subtract(SIG_PRED_NEG_NL, \
                                                SIG_NEG)), name = "Cost_nl_neg")
                cost_nl = tf.reduce_sum(tf.square(tf.subtract(SIG_PRED_NL, \
                                                SIG)), name = "Cost_nl_neg")

            '''
            --------------------------------------------------------------------------
            Define the Variables to be optimized for the specific model
            '''
            with tf.name_scope('Variable_List'):
                VARS_le = vars_le
                if bezier_applied == "Yes" and bezier_controlled == "Yes" \
                   and bezier_train_controllers == "Yes":
                    VARS_nl = vars_nl
                elif bezier_applied == "Yes" and bezier_controlled == "Yes" \
                   and bezier_train_controllers == "Yes":
                    VARS_nl = vars_nl[:-3]
                else:
                    VARS_nl = vars_nl
            '''
            --------------------------------------------------------------------------
            Define the Optimizer to minimize the cost function
            '''
            with tf.name_scope("Optimization"):
                l_rate_le = ml_settings["learn_rate_le"]
                optim_le = getattr(tf.train, ml_settings["optimizer_le"])
                optimizer_le  = optim_le(l_rate_le).minimize(train_le, var_list = VARS_le)

                l_rate_nl = ml_settings["learn_rate_nl"]
                optim_nl = getattr(tf.train, ml_settings["optimizer_nl"])
                optimizer_nl_pos  = optim_nl(l_rate_nl).minimize(train_nl_pos, var_list = VARS_nl)
                optimizer_nl_neg  = optim_nl(l_rate_nl).minimize(train_nl_neg, var_list = VARS_nl)

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
                sum_writer_nl = getattr(ConLawL.ModelVariables(), damage_model_type + "Summary")
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
            print("EPOCH STEP:", epoch_i, "\n", \
                    "-->",  "training_cost_le =", train_cost_le/eps_le_train.shape[0], '\n', \
                    "-->",  "testing_cost_le  =", test_cost_le/eps_le_test.shape[0], "\n", \
                "Trained Variables:", "E = ", vars_le_limit['E'].eval(session=sess) , "NU = ", \
                vars_le_limit['NU'].eval(session=sess))
            
            actual_tolerance = np.abs(prev_train_cost_le - train_cost_le)
            if actual_tolerance < ml_settings['learn_crit_le']:
                epoch_le = epoch_i
                break
            prev_train_cost_le = train_cost_le

        print(" OPTIMIZATION OF THE LINEAR ELASTIC VARIABLES FINISHED")
        print("    Final Tolerance of Training Cost: ", actual_tolerance)

            
        '''
        -------------------------------------------------------------------------------
        Training of the nonlinear damage model chosen in the ModelSettings.json
        -------------------------------------------------------------------------------
        '''

        print(" ------------------------------------------------------------------------------")
        print("    Damage Parameters")
        print(" ------------------------------------------------------------------------------")
    

        prev_train_cost_nl = 0.0
        randomizer_nl = np.arange(eps_nl_train.shape[0])

        #print('S1', sess.run(S_1, feed_dict={EPS: eps_nl_train, SIG: sig_nl_train}))
        #print('S2', sess.run(S_2, feed_dict={EPS: eps_nl_train, SIG: sig_nl_train}))
        #print('TauPos', sess.run(TAU_POS, feed_dict={EPS: eps_nl_train, SIG: sig_nl_train}))
        #print('TauPos', sess.run(TAU_NEG, feed_dict={EPS: eps_nl_train, SIG: sig_nl_train}))
        #print('DPos', sess.run(D_POS, feed_dict={EPS: eps_nl_train, SIG: sig_nl_train}))
        #print('DNeg', sess.run(D_NEG, feed_dict={EPS: eps_nl_train, SIG: sig_nl_train}))
        #print('SigPos', sess.run(SIG_PRED_POS_NL, feed_dict={EPS: eps_nl_train, SIG: sig_nl_train}))
        

        for epoch_i in range(n_epochs_nl):
            np.random.shuffle(randomizer_nl)
            eps_nl_rand     = eps_nl_train[randomizer_nl]
            sig_pos_nl_rand = sig_pos_nl_train[randomizer_nl]
            sig_neg_nl_rand = sig_neg_nl_train[randomizer_nl]

            for (inps1, inps2, inps3) in zip(eps_nl_rand, sig_pos_nl_rand, sig_neg_nl_rand):
                eps     = [inps1]
                sig_pos = [inps2]
                sig_neg = [inps3]
                mini_randomizer = np.random.randint(0,2)
                if mini_randomizer == 0:
                    sess.run(optimizer_nl_pos, feed_dict = {EPS: eps, SIG_POS: sig_pos})
                    sess.run(optimizer_nl_neg, feed_dict = {EPS: eps, SIG_NEG: sig_neg})
                else:
                    sess.run(optimizer_nl_neg, feed_dict = {EPS: eps, SIG_NEG: sig_neg})
                    sess.run(optimizer_nl_pos, feed_dict = {EPS: eps, SIG_POS: sig_pos})
    
            train_cost_nl = sess.run(cost_nl, feed_dict={EPS: eps_nl_train, SIG: sig_nl_train})
            test_cost_nl  = sess.run(cost_nl, feed_dict={EPS: eps_nl_test, SIG: sig_nl_test})
            summary = sess.run(merged_summaries, feed_dict={EPS: eps_nl_train, SIG: sig_nl_train})
            writer.add_summary(summary, global_step = epoch_i)            

            print("EPOCH STEP:", epoch_i, "\n", \
            "-->",  "training_cost_nl =", train_cost_nl/eps_nl_train.shape[0], '\n', \
            "-->",  "testing_cost_nl  =", test_cost_nl/eps_nl_test.shape[0], "\n", \
        "Trained Variables:",  "\n",\
            "Sp = ", vars_nl_limit['SP'].eval(session=sess), vars_nl[0].eval(session=sess), \
            "Sbi = ", vars_nl_limit['SBI'].eval(session=sess), vars_nl[1].eval(session=sess), \
            "Gc = ", vars_nl_limit['GC'].eval(session=sess), vars_nl[2].eval(session=sess), \
            "Ft = ", vars_nl_limit['FT'].eval(session=sess),  \
            "Gt = ", vars_nl_limit['GT'].eval(session=sess)) 
            
            if np.abs(prev_train_cost_nl - train_cost_nl) < ml_settings['learn_crit_nl']:
                epoch_nl = epoch_i
                break
            prev_train_cost_nl = train_cost_nl
    


    
    
    
    
                           
        
    
        
        
        

    
        
    










