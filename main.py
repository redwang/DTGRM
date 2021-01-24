#!/usr/bin/python2.7

import torch
from model import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random
import pickle
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--dataset', default="gtea")
parser.add_argument('--split', default='1')
parser.add_argument('--num_stages', type=str, default='4')
parser.add_argument('--num_layers', type=str, default='10')
parser.add_argument('--num_f_maps', type=str, default='64')
parser.add_argument('--df_size', type=str, default='3')

args = parser.parse_args()

num_stages = int(args.num_stages)
num_layers = int(args.num_layers)
num_f_maps = int(args.num_f_maps)
df_size = int(args.df_size)

features_dim = 2048
bz = 1
lr = 0.0005
num_epochs = 50

if args.dataset == "50salads":
    sample_rate = 30

if args.dataset == "gtea":
    sample_rate = 1

if args.dataset == "breakfast":
    sample_rate = 15

vid_list_file = "./data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = "./data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
features_path = "./data/"+args.dataset+"/features/"
gt_path = "./data/"+args.dataset+"/groundTruth/"

mapping_file = "./data/"+args.dataset+"/mapping.txt"

model_dir = "./models/"+args.dataset+"/split_"+args.split+"/stages_"+\
args.num_stages+"_layers_"+args.num_layers+"_fmaps_"+args.num_f_maps+"_dfsize_"+args.df_size
results_dir = "./results/"+args.dataset+"/split_"+args.split+"/stages_"+\
args.num_stages+"_layers_"+args.num_layers+"_fmaps_"+args.num_f_maps+"_dfsize_"+args.df_size
 
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
actions_dict = dict()
for a in actions:
    actions_dict[a.split()[1]] = int(a.split()[0])

num_classes = len(actions_dict)

trainer = Trainer(num_stages, num_layers, num_f_maps, df_size, features_dim, num_classes, actions_dict)
if args.action == "train":
    print("Train::"+args.dataset+"_split_"+args.split+"---stages_"+args.num_stages+
        "_layers_"+args.num_layers+"_fmaps_"+args.num_f_maps+"_dfsize_"+args.df_size)
    batch_gen = BatchGenerator(num_classes, actions_dict, gt_path, features_path, sample_rate)
    batch_gen.read_data(vid_list_file)
    trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

if args.action == "predict":
    print("Predict::"+args.dataset+"_split_"+args.split+"---stages_"+args.num_stages+
        "_layers_"+args.num_layers+"_fmaps_"+args.num_f_maps+"_dfsize_"+args.df_size)
    trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, 
        actions_dict, device, sample_rate)
