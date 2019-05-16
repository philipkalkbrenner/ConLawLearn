class ModelSettings(object):
    def GetLinearElasticModel(settings):
        name = settings["optimization_model_data"]["linear_elastic"]
        return name

    def GetLinearElasticType(settings):
        name = settings["optimization_model_data"]["linear_elastic"]["type"]
        return name

    def GetLinearElasticAssumption(settings):
        name = settings["optimization_model_data"]["linear_elastic"]['assumption']
        return name

    def GetLinearElasticModelName(settings):
        name = settings["optimization_model_data"]["linear_elastic"]['type'] + \
               settings["optimization_model_data"]["linear_elastic"]['assumption']
        return name

    def GetDamageModel(settings):
        name = settings["optimization_model_data"]["non_linear"]
        return name

    def GetDamageTypeTension(settings):
        name = settings["optimization_model_data"]["non_linear"]["damage_ten"]
        return name

    def GetDamageTypeCompression(settings):
        name = settings["optimization_model_data"]["non_linear"]["damage_comp"]
        return name

    def GetDamageModelType(settings):
        name = settings["optimization_model_data"]["non_linear"]["damage_ten"] +\
               settings["optimization_model_data"]["non_linear"]["damage_comp"]
        return name

    def GetDamageModelName(settings):
        name = settings["optimization_model_data"]["non_linear"]["yield_surf_ten"] +\
               settings["optimization_model_data"]["non_linear"]["damage_ten"] +\
               settings["optimization_model_data"]["non_linear"]["yield_surf_comp"] +\
               settings["optimization_model_data"]["non_linear"]["damage_comp"]
        return name

    
        
        
