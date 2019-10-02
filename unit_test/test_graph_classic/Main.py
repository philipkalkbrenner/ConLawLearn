import json

with open('ModelSettings.json') as m:
    model_settings = json.load(m)
with open('InitialVariableValues.json') as m:
    init_variable_values = json.load(m)

graph_name = model_settings['graph_settings']['graph_name']
graph_module = model_settings['graph_settings']['py_module']

import importlib
all_graphs = "ConLawLearn.graphs." + graph_module
Graph = importlib.import_module(all_graphs)

Graph_To_Run = getattr(Graph, graph_name)
Graph_To_Run(model_settings, init_variable_values).Run()