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

import argparse
import os

def load_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenet',help='dataset')
    parser.add_argument('--gpu_id', type=int, default=0,help='GPU ID')
    parser.add_argument('--faiss_gpu_id', type=int, default=0,help='GPU ID for FAISS, -1 for CPU')
    parser.add_argument('--kshot', type=int, default=5,help='number of exemplars')
    parser.add_argument('--cleaner', type=str, default='gcn',help='cleaner type')
    parser.add_argument('--gcnlambda', type=float, default=1.0,help='lambda hyperparameter for the GCN')
    parser.add_argument('--num_exp', type=int, default=1,help='number of episodes')
    parser.add_argument('--network', type=str, default='resnet10',help='network used')
    parser.add_argument('--valset', default=False, action='store_true',help='If True, validation split of LowShot Imagenet Benchmark is used for the evaluation.')


    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    return args
