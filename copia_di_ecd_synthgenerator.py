#!/usr/bin/env python
# coding: utf-8 

import pathlib
import pickle
import networkx as nx

from gen_model import GenerativeModel

# Hyperparameters of the Data Generators
NUM_NODES = 100
NUM_EDGES = 800
ETA = [-1., -0.5, 0.0, 0.5, 1.]  # Polarity of each generated community, each polarity must stand between -1 and +1
NUM_COMMUNITIES = len(ETA)
SOCIAL_PRIOR_SIZE = 8  # Change this value if you prefer to have a network with more "social" communities (this must always be greater than zero)
ECHO_CHAMBER_PRIOR_SIZE = 16  # Change this value if you prefer to have a network with more "polarized" communities (this must always be greater than zero)

synth_model = GenerativeModel(NUM_NODES, NUM_COMMUNITIES,
                                M=NUM_EDGES,
                                eta=ETA,
                                social_prior_size=SOCIAL_PRIOR_SIZE,
                                ec_prior_size=ECHO_CHAMBER_PRIOR_SIZE,
                                prop_prior_size=5,
                                social_beta=0.01,
                                ec_beta=1.)

synth_model.G

"""#### Save graph data on file ```/content/Echo-Chamber-Detection/src/generated_data_dir/who_to_follow_graph.gml```"""

data_dir = pathlib.Path.cwd()
data_folder_path = data_dir / 'generated_data_dir'
data_folder_path.mkdir(parents=True, exist_ok=True)

nx.write_graphml(synth_model.G, data_folder_path / 'who_to_follow_graph.gml')

"""#### Optional. Generate set of items, item polarities and user activations."""

NUM_ACTIVATIONS = 500 # Change this value if you want more or less activations

synth_model.generate_propagations(NUM_ACTIVATIONS,
                                  zipf_exponent=3.5,
                                  discard_above='N',
                                  discard_below=2,
                                  )

item_ids_list = list(range(len(synth_model.polarities)))
item_polarities = synth_model.polarities

user_activations = synth_model.item2prop

"""#### Save data on files:

*   List of ItemIDs ```/content/Echo-Chamber-Detection/src/generated_data_dir/set_of_item_IDs.pkl```
*   List of item polarities ```/content/Echo-Chamber-Detection/src/generated_data_dir/item_polarities.pkl```
*   List of user activations ```/content/Echo-Chamber-Detection/src/generated_data_dir/user_activations.pkl```

"""

with open(data_folder_path / 'set_of_item_IDs.pkl', 'wb') as file:
  pickle.dump(item_ids_list, file)

with open(data_folder_path / 'item_polarities.pkl', 'wb') as file:
  pickle.dump(item_polarities, file)

with open(data_folder_path / 'user_activations.pkl', 'wb') as file:
  pickle.dump(user_activations, file)
