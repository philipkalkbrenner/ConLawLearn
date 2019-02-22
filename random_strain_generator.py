import numpy as np

class RandomStrainGenerator(object):
    def __init__(self, unit_test_settings):
        settings = unit_test_settings
        number_of_directions = settings['number_of_directions']
        number_of_lambda     = settings['number_of_lambda']
        eps_max              = settings['maximum_strain']

        lowest = -np.pi
        uppermost = np.pi
        low_1 = -np.pi/2
        upper_1 = np.pi/2

        Theta = np.random.uniform(lowest, uppermost, number_of_directions)
        Phi = np.random.uniform(lowest, uppermost, number_of_directions)

        Theta_comp = np.random.uniform(lowest, low_1, number_of_directions)
        Phi_comp = np.random.uniform(low_1, upper_1, number_of_directions)

        lambda_ = np.random.uniform(0,eps_max, number_of_lambda)

        eps_xx_comp__ = np.einsum('i,k->ik',lambda_,np.cos(Theta_comp))
        eps_yy_comp__ = np.einsum('i,k->ik',lambda_, np.multiply(np.sin(Theta_comp),np.cos(Phi_comp)))
        eps_xy_comp__ = np.einsum('i,k->ik',lambda_, np.multiply(np.sin(Theta_comp),np.sin(Phi_comp)))

        eps_xx_comp = np.ravel(eps_xx_comp__,order='F')
        eps_yy_comp = np.ravel(eps_yy_comp__,order='F')
        eps_xy_comp = np.ravel(eps_xy_comp__,order='F')
        gamma_xy_comp = 2*eps_xy_comp

        eps_xx__ = np.einsum('i,k->ik',lambda_,np.cos(Theta))
        eps_yy__ = np.einsum('i,k->ik',lambda_, np.multiply(np.sin(Theta),np.cos(Phi)))
        eps_xy__ = np.einsum('i,k->ik',lambda_, np.multiply(np.sin(Theta),np.sin(Phi)))

        eps_xx = np.ravel(eps_xx__,order='F')
        eps_yy = np.ravel(eps_yy__,order='F')
        eps_xy = np.ravel(eps_xy__,order='F')
        gamma_xy = 2*eps_xy

        eps_xx_tot = np.concatenate((eps_xx, eps_xx_comp),axis=None)
        eps_yy_tot = np.concatenate((eps_yy, eps_yy_comp),axis=None)
        gamma_xy_tot = np.concatenate((gamma_xy, gamma_xy_comp),axis=None)

        epsilon = np.transpose(np.stack((eps_xx_tot,eps_yy_tot,gamma_xy_tot)))
        #np.random.shuffle(epsilon)
        

        self.GetStrain = epsilon

    def GetRandomStrainForPlot():
        number_of_directions = 1
        number_of_lambda     = 100
        eps_max              = 0.05

        lowest = -np.pi
        uppermost = np.pi
        low_1 = -np.pi/2
        upper_1 = np.pi/2

        Theta = np.random.uniform(lowest, uppermost, number_of_directions)
        Phi = np.random.uniform(lowest, uppermost, number_of_directions)

        Theta_comp = np.random.uniform(lowest, low_1, number_of_directions)
        Phi_comp = np.random.uniform(low_1, upper_1, number_of_directions)

        #lambda_ = np.random.uniform(0,eps_max, number_of_lambda)
        lambda_ = np.arange(0,eps_max, eps_max/number_of_lambda)
        
        '''
        eps_xx_comp__ = np.einsum('i,k->ik',lambda_,np.cos(Theta_comp))
        eps_yy_comp__ = np.einsum('i,k->ik',lambda_, np.multiply(np.sin(Theta_comp),np.cos(Phi_comp)))
        eps_xy_comp__ = np.einsum('i,k->ik',lambda_, np.multiply(np.sin(Theta_comp),np.sin(Phi_comp)))

        eps_xx_comp = np.ravel(eps_xx_comp__,order='F')
        eps_yy_comp = np.ravel(eps_yy_comp__,order='F')
        eps_xy_comp = np.ravel(eps_xy_comp__,order='F')
        gamma_xy_comp = 2*eps_xy_comp
        '''
        
        eps_xx__ = np.einsum('i,k->ik',lambda_,np.cos(Theta))
        eps_yy__ = np.einsum('i,k->ik',lambda_, np.multiply(np.sin(Theta),np.cos(Phi)))
        eps_xy__ = np.einsum('i,k->ik',lambda_, np.multiply(np.sin(Theta),np.sin(Phi)))

        eps_xx = np.ravel(eps_xx__,order='F')
        eps_yy = np.ravel(eps_yy__,order='F')
        eps_xy = np.ravel(eps_xy__,order='F')
        gamma_xy = 2*eps_xy
        '''

        eps_xx_tot = np.concatenate((eps_xx, eps_xx_comp),axis=None)
        eps_yy_tot = np.concatenate((eps_yy, eps_yy_comp),axis=None)
        gamma_xy_tot = np.concatenate((gamma_xy, gamma_xy_comp),axis=None)
        '''

        epsilon = np.transpose(np.stack((eps_xx,eps_yy,gamma_xy)))
        return epsilon

    def GetPureCompressionStrain():
        Theta = -2*np.pi/3.0
        Phi = np.pi/4.0
        eps_max              = 0.05
        number_of_lambda     = 100

        lambda_ = np.arange(0,eps_max, eps_max/number_of_lambda)

        eps_xx__ = np.multiply(lambda_, np.cos(Theta))
        eps_yy__ = np.multiply(lambda_, np.multiply(np.sin(Theta),np.cos(Phi)))
        eps_xy__ = np.multiply(lambda_, np.multiply(np.sin(Theta),np.sin(Phi)))

        #eps_xx__ = np.einsum('i,k->ik',lambda_, np.cos(Theta))
        #eps_yy__ = np.einsum('i,k->ik',lambda_, np.multiply(np.sin(Theta),np.cos(Phi)))
        #eps_xy__ = np.einsum('i,k->ik',lambda_, np.multiply(np.sin(Theta),np.sin(Phi)))

        eps_xx = np.ravel(eps_xx__,order='F')
        eps_yy = np.ravel(eps_yy__,order='F')
        eps_xy = np.ravel(eps_xy__,order='F')
        gamma_xy = 2*eps_xy
        '''

        eps_xx_tot = np.concatenate((eps_xx, eps_xx_comp),axis=None)
        eps_yy_tot = np.concatenate((eps_yy, eps_yy_comp),axis=None)
        gamma_xy_tot = np.concatenate((gamma_xy, gamma_xy_comp),axis=None)
        '''

        epsilon = np.transpose(np.stack((eps_xx,eps_yy,gamma_xy)))
        return epsilon






