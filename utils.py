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

import json
import random

import numpy as np

import pdb

def load_config(args):
    cfg = {}
    cfg['data_dir'] = 'data/'
    cfg['k'] = 50

    if args.valset:
        cfg['num_episodes'] = 1
    else:
        cfg['num_episodes'] = 5        

    # Lambda tuned according to our experiments on the validation set 
    if args.network == 'resnet10':
        if args.kshot == 1:
            cfg['gcnlambda'] = 0.01
        elif args.kshot == 2:
            cfg['gcnlambda'] = 0.05
        elif args.kshot == 5:
            cfg['gcnlambda'] = 0.5
        elif args.kshot == 10:
            cfg['gcnlambda'] = 1.0
        elif args.kshot == 20:
            cfg['gcnlambda'] = 5.0
    elif args.network == 'resnet50pca':
        if args.kshot == 1:
            cfg['gcnlambda'] = 0.01
        elif args.kshot == 2:
            cfg['gcnlambda'] = 0.05
        elif args.kshot == 5:
            cfg['gcnlambda'] = 1.0
        elif args.kshot == 10:
            cfg['gcnlambda'] = 5.0
        elif args.kshot == 20:
            cfg['gcnlambda'] = 5.0

    cfg['args'] = args
    return cfg

def load_features(cfg, split):
    fn = '{}/features/{}_{}_{}.npz'.format(cfg['data_dir'], cfg['args'].dataset, split, cfg['args'].network)
    npzfile = np.load(fn)
    features = l2_normalize_features(npzfile['features_all'])
    labels = npzfile['labels_all']

    return features, labels

def load_noisy_features(cfg):
    if cfg['args'].valset:
        fn = '{}/features/yfcc_{}_{}_val.npz'.format(cfg['data_dir'], cfg['args'].dataset, cfg['args'].network)
    else:
        fn = '{}/features/yfcc_{}_{}_test.npz'.format(cfg['data_dir'], cfg['args'].dataset, cfg['args'].network)

    npzfile = np.load(fn)
    features = l2_normalize_features(npzfile['features_all'])
    labels = npzfile['labels_all']

    return features, labels

def get_novel_class_ids(cfg):
    if cfg['args'].dataset == 'imagenet':
        split_fn = '{}/splits/imagenet_lowshot_category_splits.json'.format(cfg['data_dir'])
        with open(split_fn, 'r') as f:label_idx = json.load(f)

        if cfg['args'].valset:
            return np.asarray(label_idx['novel_classes_1'])
        else:
            return np.asarray(label_idx['novel_classes_2'])
    elif cfg['args'].dataset == 'places365':
        if cfg['args'].valset:
            split_fn = '{}/splits/places365_novel_classes_val_phase.npy'.format(cfg['data_dir'])
        else:
            split_fn = '{}/splits/places365_novel_classes_test_phase.npy'.format(cfg['data_dir'])

        return np.load(split_fn)       


def get_splits(cfg, novel_class_ids, train_labels, episode_id):
    # Splits are predefined for ImageNet by Hariharan, Ross Girshick. We use random selection with
    # fixed seed for Places365.

    if cfg['args'].dataset == 'imagenet':
        splits = np.load('{}/splits/mapped_splits_{:d}.npy'.format(cfg['data_dir'], episode_id + 1))
        novel_class_splits = splits[novel_class_ids,:cfg['args'].kshot]
    elif cfg['args'].dataset == 'places365':
        np.random.seed(episode_id)
        novel_class_splits = np.zeros((len(novel_class_ids), cfg['args'].kshot), dtype=np.int32)
        for i, class_id in enumerate(novel_class_ids):
            cur_idx = np.where(train_labels == class_id)[0]
            sel_idx = np.random.choice(cur_idx, cfg['args'].kshot, replace=False)
            novel_class_splits[i, :] = sel_idx

    return novel_class_splits.flatten()

            
def l2_normalize_features(features):
  """Normalize rows of the features matrix by their corresponding l2-norm."""
  features = features / np.linalg.norm(features, ord=2, axis=1, keepdims=True)
  # Handle cases where the norm might be 0, resulting in nan output
  features[~np.isfinite(features)] = 0.0
  return features


def get_prototypical_classifier(clean_features, clean_labels, noisy_features, noisy_labels, weights):
    label_set = np.unique(clean_labels)
    dim = clean_features.shape[1]
    classifier = np.zeros((dim, label_set.shape[0]))

    for i, cur_label in enumerate(label_set):
        cur_clean_idx = np.where(clean_labels==cur_label)[0]
        cur_noisy_idx = np.where(noisy_labels==cur_label)[0]

        cur_clean_features = clean_features[cur_clean_idx,:].T
        cur_noisy_features = noisy_features[cur_noisy_idx,:].T

        cur_noisy_weights = weights[cur_noisy_idx]

        weighted_feats = np.multiply(cur_noisy_features, cur_noisy_weights)
        all_feats = np.concatenate((cur_clean_features, weighted_feats), axis=1)

        class_proto = all_feats.mean(1)
        classifier[:, i] = class_proto

    classifier = l2_normalize_features(classifier.T).T
    return classifier, label_set

def run_eval(classifier, label_set, novel_class_ids, test_features, test_labels):
    is_novel = np.in1d(test_labels, novel_class_ids)
    test_features = test_features[is_novel,:]
    test_labels = test_labels[is_novel]
    scores = np.dot(test_features, classifier)
    topk_labels = np.argsort(-scores, 1)[:,:5]
    real_labels = label_set[topk_labels]
    is_correct = np.sum(real_labels == test_labels.reshape((-1,1)), axis=1)
    acc = is_correct.mean() * 100.

    return acc


