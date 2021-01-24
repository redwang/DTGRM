#!/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np

from layers import SingleStageModel, GCNStageModel, exchange_time

class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, df_size, dim, num_classes, actions_dict):
        super(MultiStageModel, self).__init__()
        self.actions_dict = actions_dict

        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(GCNStageModel(num_layers, num_f_maps, df_size, num_classes, num_classes)) 
            for s in range(num_stages-1)])

    def forward(self, x, mask):
        exchange_outputs = []
        exchange_labels = []
        exchange_cls_outputs = []

        ex_x, ex_label = exchange_time(x)
        out, ex_out, ex_gt, ex_pred = self.stage1(x, mask, ex_x, ex_label)
        outputs = out.unsqueeze(0)
        exchange_outputs.append(ex_out)
        exchange_labels.append(ex_gt)
        exchange_cls_outputs.append(ex_pred)
        for s in self.stages:
            out, ex_out, ex_gt, ex_pred = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask, F.softmax(ex_pred, dim=1) * mask[:, 0:1, :], ex_label)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
            exchange_outputs.append(ex_out)
            exchange_labels.append(ex_gt)
            exchange_cls_outputs.append(ex_pred)
        exchange_outputs = torch.stack(exchange_outputs)
        exchange_labels = torch.stack(exchange_labels)
        exchange_cls_outputs = torch.stack(exchange_cls_outputs)
        return outputs, exchange_outputs, exchange_labels, exchange_cls_outputs



class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, df_size, dim, num_classes, actions_dict):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, df_size, dim, num_classes, actions_dict)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                predictions, exchange_outputs, exchange_labels, exchange_cls_outputs = self.model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                for p in exchange_cls_outputs:
                    loss += 0.5*self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                for pred, gt in zip(exchange_outputs, exchange_labels):
                    loss += 2*self.ce(pred.transpose(2, 1).contiguous().view(-1, 2), gt.view(-1))

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()
            torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
                                                               float(correct)/total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                # print(vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions, _, _, _ = self.model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
