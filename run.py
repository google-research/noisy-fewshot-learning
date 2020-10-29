# Copyright 2020 Noisy-FewShot-Learning Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import numpy as np
import torch
import pdb

import cleaner
import load_args
import utils


# Parse the commandline arguments
args = load_args.load_args() 

# Create the config dictionary
cfg = utils.load_config(args)

# Get the class IDs for the novel set of classes
novel_class_ids = utils.get_novel_class_ids(cfg)

# Load the training and test set features
train_features, train_labels = utils.load_features(cfg, 'train')
val_features, val_labels = utils.load_features(cfg, 'val')

# Load the noisy set features, extracted from images retrieved from YFCC100M
noisy_features, noisy_labels = utils.load_noisy_features(cfg)

# Start running the few-shot experiments
all_acc = np.zeros((cfg['num_episodes'],)) 
for episode_id in range(cfg['num_episodes']):

	# Get the indices of clean images for this episode
	ep_indices = utils.get_splits(cfg, novel_class_ids, train_labels, episode_id)

	# Only select the features corresponding to the clean images
	ep_clean_feats = train_features[ep_indices,:]
	ep_clean_labels = train_labels[ep_indices]

	# Run the cleaner to assign relevance weights
	rel_weights = cleaner.run_cleaner(cfg, ep_clean_feats, ep_clean_labels, noisy_features, noisy_labels, faiss_gpu_id = args.faiss_gpu_id)

	# Create the prototypical classifier
	classifier, label_set = utils.get_prototypical_classifier(ep_clean_feats, ep_clean_labels, noisy_features, noisy_labels, rel_weights)

	# Classify the test images
	accuracy = utils.run_eval(classifier, label_set, novel_class_ids, val_features, val_labels)
	all_acc[episode_id] = accuracy

	print('{}, {}-shot, Episode/split:{:d}, Accuracy: {:.2f}'.format(cfg['args'].dataset, cfg['args'].kshot, episode_id, accuracy))


std_results  = all_acc.std(axis=0)
ci95_results = 1.96*std_results/np.sqrt(cfg['num_episodes'])
print('Completed {:d} episodes (splits). {}, {}-shot, Average Accuracy: {:.2f} +- {:.2f}'.format(cfg['num_episodes'], cfg['args'].dataset, cfg['args'].kshot, all_acc.mean(), ci95_results))
