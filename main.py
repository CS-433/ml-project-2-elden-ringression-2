from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import sys
import time
from collections import OrderedDict
import traceback
from sklearn.metrics import confusion_matrix
import csv
import numpy as np
import glob

# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from tqdm import tqdm

from torchlight import DictAction

# LR Scheduler
from timm.scheduler.cosine_lr import CosineLRScheduler

# stgcn
from numpy.lib.format import open_memmap
from stgcn_packages.feeder.feeder import Feeder
from torch.utils.data import DataLoader
from stgcn_packages.net.st_gcn import Model
import json



def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def import_class(import_str):
    mod_str, _sep, class_str = import_str.rpartition('.')
    __import__(mod_str)
    try:
        return getattr(sys.modules[mod_str], class_str)
    except AttributeError:
        raise ImportError('Class %s cannot be found (%s)' % (class_str, traceback.format_exception(*sys.exc_info())))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_parser():
    parser = argparse.ArgumentParser(description='SkateFormer: Skeletal-Temporal Trnasformer for Human Action Recognition')
    parser.add_argument('--work-dir', default='./work_dir', help='the work folder for storing results')
    parser.add_argument('--model_saved_name', default='')
    parser.add_argument('--config', default='./config', help='path to the configuration file')
    parser.add_argument('--model_name', default='skateformer', help='model name')

    # processor
    parser.add_argument('--phase', default='train', help='must be train or test')
    parser.add_argument('--save-score', type=str2bool, default=False, help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument('--seed', type=int, default=1, help='random seed for pytorch')
    parser.add_argument('--log-interval', type=int, default=100, help='the interval for printing messages (#iteration)')
    parser.add_argument('--save-interval', type=int, default=1, help='the interval for storing models (#iteration)')
    parser.add_argument('--save-epoch', type=int, default=30, help='the start epoch to save model (#iteration)')
    parser.add_argument('--eval-interval', type=int, default=5, help='the interval for evaluating models (#iteration)')
    parser.add_argument('--print-log', type=str2bool, default=True, help='print logging or not')
    parser.add_argument('--show-topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument('--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument('--num-worker', type=int, default=4, help='the number of worker for data loader')
    parser.add_argument('--train-feeder-args', action=DictAction, default=dict(), help='the arguments of data loader for training')
    parser.add_argument('--test-feeder-args', action=DictAction, default=dict(), help='the arguments of data loader for test')

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument('--model-args', action=DictAction, default=dict(), help='the arguments of model')
    parser.add_argument('--weights', default=None, help='the weights for network initialization')
    parser.add_argument('--ignore-weights', type=str, default=[], nargs='+', help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument('--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--min-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--warmup-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--warmup_prefix', type=bool, default=False)
    parser.add_argument('--warm_up_epoch', type=int, default=0)
    parser.add_argument('--grad-clip', type=bool, default=False)
    parser.add_argument('--grad-max', type=float, default=1.0)
    parser.add_argument('--device', type=int, default=0, nargs='+', help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='AdamW', help='type of optimizer')
    parser.add_argument('--lr-scheduler', default='cosine', help='type of learning rate scheduler')
    parser.add_argument('--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument('--batch-size', type=int, default=256, help='training batch size')
    parser.add_argument('--test-batch-size', type=int, default=256, help='test batch size')
    parser.add_argument('--start-epoch', type=int, default=0, help='start training from which epoch')
    parser.add_argument('--num-epoch', type=int, default=80, help='stop training in which epoch')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='weight decay for optimizer')
    parser.add_argument('--lr-ratio', type=float, default=0.001, help='decay rate for learning rate')
    parser.add_argument('--lr-decay-rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--loss-type', type=str, default='CE')
    return parser


class Processor():
    def __init__(self, arg):
        self.arg = arg
        self.save_arg()

        if arg.phase == 'train':
            if not arg.train_feeder_args['debug']:
                arg.model_saved_name = os.path.join(arg.work_dir, 'runs')
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')

        self.global_step = 0
        self.load_model()
        self.load_data()

        if self.arg.phase == 'train':
            self.load_optimizer()
            self.load_scheduler(len(self.data_loader['train']))

        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_acc_epoch = 0
        self.model = self.model.cuda(self.output_device)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=self.output_device)

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        print(Model)
        self.model = Model(**self.arg.model_args)
        if self.arg.loss_type == 'CE':
            self.loss = nn.CrossEntropyLoss().cuda(output_device)
        else:
            self.loss = LabelSmoothingCrossEntropy(smoothing=0.1).cuda(output_device)

        if self.arg.weights:
            #self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict([[k.split('module.')[-1], v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def load_scheduler(self, n_iter_per_epoch):
        num_steps = int(self.arg.num_epoch * n_iter_per_epoch)
        warmup_steps = int(self.arg.warm_up_epoch * n_iter_per_epoch)

        self.lr_scheduler = None
        if self.arg.lr_scheduler == 'cosine':
            self.lr_scheduler = CosineLRScheduler(
                self.optimizer,
                t_initial=(num_steps - warmup_steps) if self.arg.warmup_prefix else num_steps,
                lr_min=self.arg.min_lr,
                warmup_lr_init=self.arg.warmup_lr,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
                warmup_prefix=self.arg.warmup_prefix,
            )
        else:
            raise ValueError()

    def save_arg(self):
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            f.write(f"# command line: {' '.join(sys.argv)}\n\n")
            yaml.dump(arg_dict, f)

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def train(self, epoch, save_model=False):
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']

        loss_value = []
        acc_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)

        for batch_idx, (data, label, index) in enumerate(process):
            self.lr_scheduler.step_update(self.global_step)
            self.global_step += 1
            with torch.no_grad():
                data = data.float().cuda(self.output_device)
                label = label.long().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # forward
            output = self.model(data)
            loss = self.loss(output, label)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            if self.arg.grad_clip:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.arg.grad_max)
            self.optimizer.step()

            loss_value.append(loss.data.item())
            timer['model'] += self.split_time()

            value, predict_label = torch.max(output.data, 1)
            acc = torch.mean((predict_label == label.data).float())
            acc_value.append(acc.data.item())
            self.train_writer.add_scalar('acc', acc, self.global_step)
            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)
            timer['statistics'] += self.split_time()

        # statistics of time consumption and loss
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean training acc: {:.2f}%.'.format(np.mean(loss_value),
                                                                                np.mean(acc_value) * 100))
        self.print_log('\tLearning Rate: {:.4f}'.format(self.lr))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
            state_dict = self.model.state_dict()
            weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])

            torch.save(weights,
                       self.arg.model_saved_name + '-' + str(epoch + 1) + '-' + str(int(self.global_step)) + '.pt')

    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None, result_file=None):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'w')
        self.model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            label_list = []
            pred_list = []
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data, label, index) in enumerate(process):
                label_list.append(label)
                with torch.no_grad():
                    data = data.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)
                    output = self.model(data)
                    loss = self.loss(output, label)
                    score_frag.append(output.data.cpu().numpy())
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    pred_list.append(predict_label.data.cpu().numpy())
                    step += 1

                if wrong_file is not None or result_file is not None:
                    predict = list(predict_label.cpu().numpy())
                    true = list(label.data.cpu().numpy())
                    for i, x in enumerate(predict):
                        if result_file is not None:
                            f_r.write(str(x) + ',' + str(true[i]) + '\n')
                        if x != true[i] and wrong_file is not None:
                            f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            score = np.concatenate(score_frag)
            loss = np.mean(loss_value)
            if 'ucla' in self.arg.feeder:
                self.data_loader[ln].dataset.sample_name = np.arange(len(score))
            accuracy = self.data_loader[ln].dataset.top_k(score, 1)
            if accuracy > self.best_acc:
                self.best_acc = accuracy
                self.best_acc_epoch = epoch + 1

            print('Accuracy: ', accuracy, ' model: ', self.arg.model_saved_name)
            if self.arg.phase == 'train':
                self.val_writer.add_scalar('loss', loss, self.global_step)
                self.val_writer.add_scalar('acc', accuracy, self.global_step)

            score_dict = dict(
                zip(self.data_loader[ln].dataset.sample_name, score))
            self.print_log('\tMean {} loss of {} batches: {}.'.format(
                ln, len(self.data_loader[ln]), np.mean(loss_value)))
            for k in self.arg.show_topk:
                self.print_log('\tTop{}: {:.2f}%'.format(
                    k, 100 * self.data_loader[ln].dataset.top_k(score, k)))

            if save_score:
                with open('{}/epoch{}_{}_score.pkl'.format(
                        self.arg.work_dir, epoch + 1, ln), 'wb') as f:
                    pickle.dump(score_dict, f)

            # acc for each class:
            label_list = np.concatenate(label_list)
            pred_list = np.concatenate(pred_list)
            confusion = confusion_matrix(label_list, pred_list)
            list_diag = np.diag(confusion)
            list_raw_sum = np.sum(confusion, axis=1)
            each_acc = list_diag / list_raw_sum
            with open('{}/epoch{}_{}_each_class_acc.csv'.format(self.arg.work_dir, epoch + 1, ln), 'w') as f:
                writer = csv.writer(f)
                writer.writerow(each_acc)
                writer.writerows(confusion)

    def export(self):
        ex_model = self.model.to('cuda:0')
        ex_model.eval()

        x_input = torch.rand(3, 64, 16, 1)
        data = torch.tensor(x_input).unsqueeze(0)
        data = data.float().to("cuda:0")
        traced_script_model = torch.jit.trace(ex_model, data)
        traced_script_model.save('torchscript.pt')

    def start(self):
        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size

            def count_parameters(model):
                return sum(p.numel() for p in model.parameters() if p.requires_grad)

            self.print_log(f'# Parameters: {count_parameters(self.model)}')
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if epoch + 1 < self.arg.num_epoch * 0.9:
                    self.train(epoch, save_model=False)
                else:
                    self.train(epoch, save_model=True)
                    self.eval(epoch, save_score=True, loader_name=['test'])

            # test the best model
            weights_path = glob.glob(os.path.join(self.arg.work_dir, 'runs-' + str(self.best_acc_epoch) + '*'))[0]
            weights = torch.load(weights_path)
            if type(self.arg.device) is list:
                if len(self.arg.device) > 1:
                    weights = OrderedDict([['module.' + k, v.cuda(self.output_device)] for k, v in weights.items()])
            self.model.load_state_dict(weights)

            wf = weights_path.replace('.pt', '_wrong.txt')
            rf = weights_path.replace('.pt', '_right.txt')
            self.arg.print_log = False
            self.eval(epoch=0, save_score=True, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.arg.print_log = True

            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            self.print_log(f'Best accuracy: {self.best_acc}')
            self.print_log(f'Epoch number: {self.best_acc_epoch}')
            self.print_log(f'Model name: {self.arg.work_dir}')
            self.print_log(f'Model total number of params: {num_params}')
            self.print_log(f'Weight decay: {self.arg.weight_decay}')
            self.print_log(f'Base LR: {self.arg.base_lr}')
            self.print_log(f'Batch Size: {self.arg.batch_size}')
            self.print_log(f'Test Batch Size: {self.arg.test_batch_size}')
            self.print_log(f'seed: {self.arg.seed}')

        elif self.arg.phase == 'test':
            wf = self.arg.weights.replace('.pt', '_wrong.txt')
            rf = self.arg.weights.replace('.pt', '_right.txt')

            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')

        elif self.arg.phase == 'export':
            self.export()

def gendata(data_path, label_path, data_out_path, label_out_path, num_person_in=1, num_person_out=1, max_frame=100):
    """
    Convert JSON-based train/val/test data into `.npy` and `.pkl` formats.
    Args:
        data_path: Path to the folder containing JSON files.
        label_path: Path to the label JSON file.
        data_out_path: Output path for the `.npy` file.
        label_out_path: Output path for the `.pkl` file.
        num_person_in: Maximum number of persons in the frame (default 1).
        num_person_out: Number of persons to include in output (default 1).
        max_frame: Maximum number of frames to include per sample (default 100).
    """
    # Load label file
    with open(label_path, 'r') as f:
        label_data = json.load(f)

    sample_name = [entry['file_name'] for entry in label_data]
    sample_label = [entry['label_index'] for entry in label_data]

    # Initialize memmap for the data
    fp = open_memmap(
        data_out_path,
        dtype='float32',
        mode='w+',
        shape=(len(sample_name), 3, max_frame, 17, num_person_out)
    )

    for i, file_name in enumerate(sample_name):
        file_path = os.path.join(data_path, file_name)
        with open(file_path, 'r') as f:
            json_data = json.load(f)

        # Extract annotations
        annotations = json_data["annotations"]

        # Initialize data array
        data = np.zeros((3, max_frame, 17, num_person_in), dtype=np.float32)

        for frame in annotations:
            frame_idx = frame["frame_index"]
            keypoints = np.array(frame["keypoints"])  # Shape: (17, 3)
            data[:3, frame_idx, :, 0] = keypoints.T  # Transpose to match (C, T, V, M)

        fp[i, :, :data.shape[1], :, :] = data[:, :max_frame, :, :num_person_out]

    # Save labels
    with open(label_out_path, 'wb') as f:
        pickle.dump((sample_name, sample_label), f)

    print(f"Saved data to {data_out_path} and labels to {label_out_path}")

if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()

    if p.model_name == 'skateformer':
        print("Using Skateformer.")
        if p.config is not None:
            with open(p.config, 'r') as f:
                default_arg = yaml.load(f)
            key = vars(p).keys()
            for k in default_arg.keys():
                if k not in key:
                    print('WRONG ARG: {}'.format(k))
                    assert (k in key)
            parser.set_defaults(**default_arg)

        arg = parser.parse_args()
        init_seed(arg.seed)
        processor = Processor(arg)
        processor.start()

    elif p.model_name == 'stgcn':
        print("Using STGCN.")

        frames_length = 64

        training_config = {
            'random_choose': True,
            'random_move': True,
            'window_size': frames_length,
            'batch_size': 16,
            'test_batch_size': 16,
            'base_lr': 0.001,
            'step': [80],
            'num_epoch': 100,
            'device': [0],
            'num_class': 5,
            'in_channels': 3,
            'edge_importance_weighting': True,
            'layout': 'my_pose',
            'strategy': 'spatial'
        }

        train_data_path = './data/assistive_furniture/frame_64/train_data.npy'
        train_label_path = './data/assistive_furniture/frame_64/train_label.pkl'
        val_data_path = './data/assistive_furniture/frame_64/val_data.npy'
        val_label_path = './data/assistive_furniture/frame_64/val_label.pkl'

        # Set paths and process datasets
        data_root = "./data/assistive_furniture/frame_64"
        parts = ['train', 'val', 'test']

        for part in parts:
            data_path = os.path.join(data_root, f"kinetics_{part}")
            label_path = os.path.join(data_root, f"kinetics_{part}_label.json")
            data_out_path = os.path.join(data_root, f"{part}_data.npy")
            label_out_path = os.path.join(data_root, f"{part}_label.pkl")

            print(f"[INFO] Processing {part} set...")
            gendata(data_path, label_path, data_out_path, label_out_path)
            print(f"[INFO] Finished processing {part} set.")

        # Initialize Feeder instances for training and validation
        train_feeder = Feeder(
            data_path=train_data_path,
            label_path=train_label_path,
            random_choose=training_config['random_choose'],
            random_move=training_config['random_move'],
            window_size=training_config['window_size']
        )

        val_feeder = Feeder(
            data_path=val_data_path,
            label_path=val_label_path,
            random_choose=False,
            random_move=False,
            window_size=training_config['window_size']
        )

        # Create DataLoaders
        train_loader = DataLoader(train_feeder, batch_size=training_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_feeder, batch_size=training_config['test_batch_size'], shuffle=False)

        print(f"Train Loader: {len(train_loader)} batches")
        print(f"Validation Loader: {len(val_loader)} batches")

        model = Model(
            in_channels=training_config['in_channels'],
            num_class=training_config['num_class'],
            edge_importance_weighting=training_config['edge_importance_weighting'],
            graph_args={
                'layout': training_config['layout'],
                'strategy': training_config['strategy']
            }
        )
        model = model.to(
            torch.device(f'cuda:{training_config["device"][0]}' if torch.cuda.is_available() else 'cpu'))

        optimizer = torch.optim.SGD(model.parameters(), lr=training_config['base_lr'], momentum=0.9,
                                    weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=training_config['step'], gamma=0.1)

        for epoch in range(training_config['num_epoch']):
            model.train()
            total_loss = 0

            for batch_data, batch_label in train_loader:
                optimizer.zero_grad()

                batch_data = batch_data.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                batch_label = batch_label.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

                output = model(batch_data)
                loss = torch.nn.CrossEntropyLoss()(output, batch_label)

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            print(f'Epoch {epoch + 1}/{training_config["num_epoch"]} - Loss: {total_loss:.4f}')
            scheduler.step()

            # Evaluate on validation set
            model.eval()
            val_loss = 0
            correct = 0
            with torch.no_grad():
                for batch_data, batch_label in val_loader:
                    batch_data = batch_data.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
                    batch_label = batch_label.to(torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

                    output = model(batch_data)
                    val_loss += torch.nn.CrossEntropyLoss()(output, batch_label).item()
                    correct += (torch.argmax(output, dim=1) == batch_label).sum().item()

            accuracy = correct / len(val_loader.dataset)
            print(f'Validation Loss: {val_loss:.4f}, Accuracy: {accuracy:.4f}')

            save_dir = './trained_models'
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, 'st_gcn_model_64.pth')
            torch.save(model.state_dict(), save_path)

            print(f"Model saved to {save_path}")

            # Set paths
            model_path = './trained_models/st_gcn_model_64.pth'
            test_data_path = './data/assistive_furniture/frame_64/test_data.npy'
            test_label_path = './data/assistive_furniture/frame_64/test_label.pkl'

            # Load the trained model
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

            # Number of classes (adjust based on your dataset)
            Gesture_names = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5']
            num_classes = len(Gesture_names)

            # Initialize the model
            model = Model(
                in_channels=3,
                num_class=num_classes,
                edge_importance_weighting=True,
                graph_args={
                    'layout': 'my_pose',
                    'strategy': 'spatial'
                }
            ).to(device)

            # Load the trained weights safely
            weights = torch.load(model_path, map_location=device)
            model.load_state_dict(weights)
            model.eval()

            print(f"Model loaded successfully from {model_path}")

            # Load test data
            test_data = np.load(test_data_path)
            with open(test_label_path, 'rb') as f:
                sample_names, test_labels = pickle.load(f)

            print(f"Test data and labels loaded successfully. Number of samples: {len(test_data)}")