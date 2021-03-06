import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from ConLawLearn import ConLawL


class GraphLinearElastic(object):
    def __init__(self, model_settings, initial_variable_values):
        self.model_settings = model_settings
        self.init_var_values = initial_variable_values

        self.input_settings = self.model_settings["input_settings"]
        self.ml_settings = self.model_settings["machine_learning_settings"]
        self.post_settings = self.model_settings["post_settings"]

    def Run(self):
        self._PrintHeaderMain()

        # Initialize Models to Train
        self._InitializeLinearModel()
        #self._InitializeDamageModel()
        self._PrintInfoModel()

        # Import the Input Data
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

        # Initialize the Graph session:
        self._InitializeSession()
        
        # Start training loop of linear model
        self._PrintInfoStartOptimizationLinear()
        self._RunTrainingLoopLinearModel()


        fig1 = self._ResultPlot(0, self.eps_le_train, self.sig_le_train, self.sig_le_eval, self.sig_le_prev)
        fig2 = self._ResultPlot(1, self.eps_le_train, self.sig_le_train, self.sig_le_eval, self.sig_le_prev)
        fig3 = self._ResultPlot(2, self.eps_le_train, self.sig_le_train, self.sig_le_eval, self.sig_le_prev)
        plt.show()

    '''
    ------------------------------------------------------------------------------
    '''


    def _InitializeLinearModel(self):
        le_model = ConLawL.ModelSettings.GetLinearElasticModel(self.model_settings)
        self.le_model_type = ConLawL.ModelSettings.GetLinearElasticType(self.model_settings)
        self.le_model_name = ConLawL.ModelSettings.GetLinearElasticModelName(self.model_settings)

    def _InitializeDamageModel(self):
        damage_model = ConLawL.ModelSettings.GetDamageModel(self.model_settings)
        self.damage_model_type =  ConLawL.ModelSettings.GetDamageModelType(self.model_settings)
        self.damage_model_name = ConLawL.ModelSettings.GetDamageModelName(self.model_settings)

        self.bezier_applied = damage_model["bezier_settings"]["applied?"]
        self.bezier_energy_approach = damage_model["bezier_settings"]["comp_energy_approach"]
        self.bezier_train_controllers = damage_model["bezier_settings"]["train_controllers"]

    def _ImportTrainingData(self):
        Inputs = ConLawL.TrainingInput(self.input_settings)
        self.eps_le = Inputs.GetStrainsLinearElastic
        self.sig_le = Inputs.GetStressesLinearElastic
        #self.eps_nl = Inputs.GetStrainsNonlinear
        #self.sig_nl = Inputs.GetStressesNonlinear

        self.eps_le_train, self.eps_le_test, self.sig_le_train, self.sig_le_test = \
                              Inputs.SplitTrainingAndTesting(self.eps_le, self.sig_le)
        #self.eps_nl_train, self.eps_nl_test, self.sig_nl_train, self.sig_nl_test = \
        #                      Inputs.SplitTrainingAndTesting(self.eps_nl, self.sig_nl)
    
    def _BuildPlaceholders(self):
        with tf.name_scope("Placeholders"):
            self.EPS = tf.placeholder(tf.float32, name="EPSILON")
            self.SIG = tf.placeholder(tf.float32, name="SIGMA")

    def _BuildVariableList(self):
        with tf.name_scope("Variables"):
            with tf.name_scope("LinearElasticModelVariables"):
                var_type_le = getattr(ConLawL.ModelVariables(), self.le_model_type)
                self.vars_le = var_type_le(self.init_var_values).Variables
                self.vars_le_plot = var_type_le(self.init_var_values).Vars4Print
                self.vars_le_limit = var_type_le.ConstrainVariables(self.vars_le, self.init_var_values)
            
    def _CallPredictedStresses(self):
        with tf.name_scope("LinearElasticLaw"):
            le_model = getattr(ConLawL.ConstitutiveLaw(), self.le_model_name)
            with tf.name_scope('PredictedStress'):
                self.SIG_PRED_LE = le_model.GetStress(self.EPS, self.vars_le_limit)

    def _ConstructCostFunction(self):
        with tf.name_scope('TrainingFunctions'):
            self.train_le = tf.reduce_sum(tf.square(tf.subtract(self.SIG_PRED_LE, \
                                            self.SIG)), name = "Cost_le")

    def _ConstructOptimizer(self):
        with tf.name_scope("Optimization"):
            self.l_rate_le = self.ml_settings["learn_rate_le"]
            optim_le = getattr(tf.train, self.ml_settings["optimizer_le"])
            self.optimizer_le  = optim_le(self.l_rate_le).minimize(self.train_le, var_list = self.vars_le)
    
    def _ConstructSummaryWriter(self):
        with tf.name_scope('AllSummaries'):
            sum_writer_le = getattr(ConLawL.ModelVariables(), self.le_model_type + "Summary")
            sum_writer_le(self.vars_le_limit)
    
    def _GlobalOpGraphStart(self):
        with tf.name_scope("GlobalOps"):
            self.init = tf.global_variables_initializer()
            self.merged_summaries = tf.summary.merge_all()

    def _InitializeSession(self):
        log_dir = self.post_settings["tensorboard_logdir"]
        self.sess = tf.Session(graph=self.graph)
        self.writer = tf.summary.FileWriter(log_dir, self.graph)
        self.sess.run(self.init)
    
    def _RunTrainingLoopLinearModel(self):
        n_epochs_le = self.ml_settings["max_epoch_le"]
    
        prev_train_cost_le = 0.0
        eps_le_prev = self.eps_le_train
        self.sig_le_prev = self.sess.run(self.SIG_PRED_LE, feed_dict={self.EPS:self.eps_le_train})
        randomizer_le = np.arange(self.eps_le_train.shape[0])

        print(" ------------------------------------------------------------------------------")
        print("       Initial Variable Value:")
        print("          ", self.sess.run(self.vars_le_limit))
        print(" ------------------------------------------------------------------------------")

        for epoch_i in range(n_epochs_le):
            np.random.shuffle(randomizer_le)
            eps_le_rand = self.eps_le_train[randomizer_le]
            sig_le_rand = self.sig_le_train[randomizer_le]

            for (inps1, inps2) in zip(eps_le_rand, sig_le_rand):
                eps = [inps1]
                sig = [inps2]
                self.sess.run(self.optimizer_le, feed_dict = {self.EPS:eps, self.SIG:sig})
            train_cost_le = self.sess.run(self.train_le, feed_dict={self.EPS: self.eps_le_train, self.SIG: self.sig_le_train})
            test_cost_le  = self.sess.run(self.train_le, feed_dict={self.EPS: self.eps_le_test, self.SIG: self.sig_le_test})
            summary = self.sess.run(self.merged_summaries, feed_dict={self.EPS: self.eps_le_train, self.SIG: self.sig_le_train})
            self.writer.add_summary(summary, global_step = epoch_i)

            self.sig_le_eval = self.sess.run(self.SIG_PRED_LE, feed_dict={self.EPS:eps_le_prev})

            print("    EPOCH STEP:", epoch_i+1, "\n", \
                  "       Training Cost =", round(train_cost_le/self.eps_le_train.shape[0], 8), '\n', \
                  "       Testing Cost  =", round(test_cost_le/self.eps_le_test.shape[0],8))
            print("       Optimized Variable Value:")
            print("          ", self.sess.run(self.vars_le_limit))
            print(" ------------------------------------------------------------------------------")
            
            actual_tolerance = np.abs(prev_train_cost_le - train_cost_le)
            if actual_tolerance < self.ml_settings['learn_crit_le']:
                epoch_le = epoch_i
                break
            prev_train_cost_le = train_cost_le

        print(" OPTIMIZATION OF THE LINEAR ELASTIC VARIABLES FINISHED")
        print("    Final Tolerance of Training Cost: ", actual_tolerance)

        
    def _ResultPlot(self, component, eps_prev, sig_train, sig_eval, sig_prev):
        if component == 0:
            add = "_xx"
        elif component == 1:
            add = "_yy"
        elif component == 2:
            add = "_xy"
        else:
            print("This is 2D, the component cannot be bigger than 2")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(eps_prev[:,component], sig_train[:,component], s=1, label = "input model")#, color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax.scatter(eps_prev[:,component], sig_eval[:,component],s=1, label = "optimized model")#, color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax.scatter(eps_prev[:,component], sig_prev[:,component],s=1, label = "initial model")#, color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax.set_title('Result Plots of Optimization')
        ax.set_xlabel('strain' r'$\ \epsilon\ $' + add + '[-]')
        ax.set_ylabel('stress' r'$\ \sigma\ $' + add + '[N/mm^2]')
        ax.legend(loc = 'lower left')


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
        print(" MODEL TO PREDICT:")
        print("    Linear Elasticity Theory: ","<<", self.le_model_name, ">>")
        print("------------------------------------------------------------------------------") 

    
    def _PrintInfoTrainingData(self):
        print(" Total Input Data = ", int(self.input_settings["train_test_factor"]*100),"%  Training Data + ",  \
              100-int(self.input_settings["train_test_factor"]*100), "%  Testing Data")
        print("    Linear Elastic:", "\n", \
              "        Training Set Size : ", "<<", self.eps_le_train.shape[0],  ">>", "States", "\n", \
              "        Testing Set Size  : ", "<<", self.eps_le_test.shape[0], ">>", "States")
        print(" ------------------------------------------------------------------------------")

    
    def _PrintInfoGraphType(self):
        print(" GRAPH CONSTRUCTION TYPE")
        print("    Linear Elastic Parameters:",  "\n",\
              "      --> Feeding Linear Input Strains", "\n",\
              "      --> Predicting Linear Stresses", "\n",\
              "      --> Loss Function with Input Stresses", "\n",\
              "      --> Optimize Linear Parameters to Minimize Loss")
        print(" ------------------------------------------------------------------------------")

    
    def _PrintInfoVariableList(self):
        print(" VARIABLES TO OPTIMIZE")
        print(" Total Number of Variables =", len(self.vars_le_plot))
        print("    Linear Elastic Variables")
        for i in range(len(self.vars_le_plot)):
            print("       -->", self.vars_le_plot[i])
        print(" ------------------------------------------------------------------------------")
    
    def _PrintInfoOptimizer(self):
        print(" GRADIENT OPTIMIZATION")
        print("    Linear Elastic Optimization: ")
        print("        Optimizer:           ", self.ml_settings["optimizer_le"], '\n',\
              "       Learning Rate:       ", self.l_rate_le, '\n',\
              "       max number of epochs:", self.ml_settings["max_epoch_le"])
        print(" ------------------------------------------------------------------------------")
    
    def _PrintInfoStartOptimizationLinear(self):
        print(" ------------------------------------------------------------------------------")
        print(" OPTIMIZATION STARTS")
        print(" ------------------------------------------------------------------------------")
        print("    Linear Elastic Parameters")
        print(" ------------------------------------------------------------------------------")

        





        

















        '''
            Final Comparison Plot
        '''
        '''
        fig1 = plt.figure()
        ax1 = fig1.add_subplot(111)
        ax1.scatter(self.eps_le_prev[:,0], self.sig_le_train[:,0])#, color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax1.scatter(self.eps_le_prev[:,0], self.sig_le_eval[:,0])#, color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax1.scatter(self.eps_le_prev[:,0], self.sig_le_prev[:,0])#, color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax1.set_title('Plot of Test Data')
        ax1.set_xlabel('stress' r'$\ \epsilon_{xx}\ [-]$')
        ax1.set_ylabel('stress' r'$\ \sigma_{xx}\ [N/mm^2]$')
        ax1.legend(loc = 'lower left')

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)
        ax2.plot(self.eps_le_prev[:,1], self.sig_le_train[:,1], color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax2.plot(self.eps_le_prev[:,1], self.sig_le_eval[:,1], color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax2.plot(self.eps_le_prev[:,1], self.sig_le_prev[:,1], color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax2.set_title('Plot of Test Data')
        ax2.set_xlabel('stress' r'$\ \epsilon_{yy}\ [-]$')
        ax2.set_ylabel('stress' r'$\ \sigma_{yy}\ [N/mm^2]$')
        ax2.legend(loc = 'lower left')

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111)
        ax3.plot(self.eps_le_prev[:,2], self.sig_le_train[:,2], color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax3.plot(self.eps_le_prev[:,2], self.sig_le_eval[:,2], color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax3.plot(self.eps_le_prev[:,2], self.sig_le_prev[:,2], color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax3.set_title('Plot of Test Data')
        ax3.set_xlabel('stress' r'$\ \gamma_{xy}\ [-]$')
        ax3.set_ylabel('stress' r'$\ \sigma_{xy}\ [N/mm^2]$')
        ax3.legend(loc = 'lower left')

        plt.show()


        fig4 = plt.figure()
        ax4 = fig4.add_subplot(111)
        ax4.scatter(self.eps_nl_train[:,0], self.sig_nl_train[:,0],s=1, label = "input model")#, color = 'gray', linewidth = 1, linestyle = '-')
        ax4.scatter(self.eps_nl_train[:,0], sig_nl_eval[:,0], s=1, label = "optimized model")#, color = 'black',  linewidth = 2, linestyle = '-.')
        ax4.scatter(self.eps_nl_train[:,0], sig_nl_prev[:,0], s=1, label = "initial model")#, color = 'black',  linewidth = 1, linestyle = '-')
        ax4.set_title('Plot of Test Data')
        ax4.set_xlabel('stress' r'$\ \epsilon_{xx}\ [-]$')
        ax4.set_ylabel('stress' r'$\ \sigma_{xx}\ [N/mm^2]$')
        ax4.legend(loc = 'lower left')

        fig5 = plt.figure()
        ax5 = fig5.add_subplot(111)
        ax5.scatter(self.eps_nl_train[:,1], self.sig_nl_train[:,1], s=1, label = "input model")#, color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax5.scatter(self.eps_nl_train[:,1], sig_nl_eval[:,1], s=1, label = "optimized model")#, color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax5.scatter(self.eps_nl_train[:,1], sig_nl_prev[:,1], s=1, label = "initial model")#, color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax5.set_title('Plot of Test Data')
        ax5.set_xlabel('stress' r'$\ \epsilon_{yy}\ [-]$')
        ax5.set_ylabel('stress' r'$\ \sigma_{yy}\ [N/mm^2]$')
        ax5.legend(loc = 'lower left')

        fig6 = plt.figure()
        ax6 = fig6.add_subplot(111)
        ax6.scatter(self.eps_nl_train[:,2], self.sig_nl_train[:,2], s=1, label = "input model")#, color = 'gray', linewidth = 1, linestyle = '-', label = "input model")
        ax6.scatter(self.eps_nl_train[:,2], sig_nl_eval[:,2], s=1, label = "optimized model")#, color = 'black',  linewidth = 2, linestyle = '-.', label = "optimized model")
        ax6.scatter(self.eps_nl_train[:,2], sig_nl_prev[:,2], s=1, label =  "initial model")#, color = 'black',  linewidth = 1, linestyle = '-', label = "initial model")
        ax6.set_title('Plot of Test Data')
        ax6.set_xlabel('stress' r'$\ \gamma_{xy}\ [-]$')
        ax6.set_ylabel('stress' r'$\ \sigma_{xy}\ [N/mm^2]$')
        ax6.legend(loc = 'lower left')

        plt.show()'''
        


    
    
    
    
                           
        
    
        
        
        

    
        
    










