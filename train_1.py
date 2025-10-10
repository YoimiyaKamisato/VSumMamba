import argparse
import os
from tqdm import tqdm
import time
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import tensorboardX
import numpy as np
from VSumMamba.metrics import Metric
from VSumMamba.build_dataloader import build_dataloader
from VSumMamba.build_augment import build_dataloader1
from VSumMamba.build_model import build_model
from VSumMamba.build_optimizer import build_optimizer
from VSumMamba.eval import select_keyshots, select_ks
import math
import pandas as pd
import h5py
from VSumMamba.utils import (
    adjust_learning_rate,
    save_model,
    load_model,
    resume_model,
)

pd_epoch = []
pd_batch_size = []
pd_lr = []
pd_runtime = []
pd_loss = []
pd_F_measure_k = []
pd_kendall_k = []
pd_spearman_k = []
pd_test_loss = []
pd_kendall_score=[]
pd_spearman_score=[]
t_xum=""
for i in range(7999,12000):
    t_xum=t_xum+str(i+1)
    if  i<11999:
        t_xum=t_xum+","




def parse_args():
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument(
        '--roundtimes', type=str, default='1', help='Roundtimes.'
    )
    parser.add_argument(
        '--dataset', default='TVSum_77', help='Dataset names.'
    )
    parser.add_argument(
        '--test_dataset',
        type=str,
        default="1,2,11,16,18,20,31,32,35,46",
        help='The number of test video in the dataset.',

    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=2,
        help='The number of classes in the dataset.',
    )
    parser.add_argument(
        '--lamda',
        type=float,
        default=0.1,
        help='The number of sequence.',
    )
    parser.add_argument(
        '--sequence',
        type=int,
        default=16,
        help='The number of sequence.',
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=40,
        help='input batch size for training',
    )
    parser.add_argument(
        '--val_batch_size',
        type=int,
        default=40,
        help='input batch size for val',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        # default = 100,
        default=100,
        help='number of epochs to train'
    )
    parser.add_argument(
        '--test_epochs',
        type=int,
        default=1,
        help='number of internal epochs to test',
    )
    parser.add_argument('--optim', default='sgd', help='Model names.')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='beta1')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='beta2')
    parser.add_argument('--eps', type=float, default=1e-8, help='eps')
    parser.add_argument('--lr',
                        type=float,
                     
                        default=0.0001,

                        help='learning rate')
    parser.add_argument(
        '--warmup_epochs',
        type=float,
        default=10,
        help='number of warmup epochs',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.9, help='SGD momentum'
    )
    parser.add_argument(
        '--weight_decay', type=float,
        default=0.00008
    )
    parser.add_argument(
        '--nesterov',
        action='store_true',
        default=False,
        # default=True,
        help='To use nesterov or not.',
    )
    parser.add_argument(
        '--no_cuda',
        action='store_true',
        default=False,
        help='disables CUDA training',
    )
    parser.add_argument(
        '--lr_scheduler',
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help='how to schedule learning rate',
    )
    parser.add_argument(
        '--resume', action='store_true', default=False, help='Resume training'
    )
    parser.add_argument(
        '--gpu_id', default='0', type=str, help='id(s) for CUDA_VISIBLE_DEVICES'
    )

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    return args

def val(model, val_loader, criterion,criterion_mse, epoch, args):

    
    model.eval()
    if epoch == -1:
        epoch = args.epochs - 1
    test_loss_1 = Metric('test_loss_1')
    test_loss_2 = Metric('test_loss_2')
    test_loss_all = Metric('test_loss_all')
    global pd_F_measure_k
    global pd_kendall_k
    global pd_spearman_k
    global pd_kendall_score
    global pd_spearman_score
    global pd_test_loss
    with tqdm(
            total=len(val_loader), desc='Validate Epoch #{}'.format(epoch + 1)
    ) as t:
        with torch.no_grad():
            predicted_multi_list = []
            target_multi_list = []
            video_number_list = []
            image_number_list = []
            for data, target, video_number, image_number in val_loader:
                predicted_list = []
                target_list = []
                if args.cuda:
                    data = data.cuda()

                multi_test_loss = 0
                output = model(data)
                multi_target = target.permute(1, 0)
                video_number = video_number
                image_number = image_number
                multi_output = output
                for sequence in range(args.sequence):
                    target = multi_target[sequence].cuda()
                    output = multi_output[sequence]
                    loss = criterion(output, target)
                    multi_test_loss += loss

                    predicted_ver2 = []
                    sigmoid = nn.Sigmoid()
                    outputs_sigmoid = sigmoid(output)
                    for s in outputs_sigmoid:
                        predicted_ver2.append(float(s[1]))
                    predicted_list.append(predicted_ver2)
                    target_list.append(target.tolist())
                multi_test_loss /= args.sequence
                t.update(1)
                predicted_list = torch.Tensor(predicted_list).permute(1, 0)
                predicted_list = torch.Tensor(predicted_list).reshape(args.val_batch_size * args.sequence)
                target_list = torch.Tensor(target_list).permute(1, 0)
                target_list = torch.Tensor(target_list).reshape(args.val_batch_size * args.sequence)
                video_number = video_number.reshape(args.val_batch_size * args.sequence)
                image_number = image_number.reshape(args.val_batch_size * args.sequence)
                predicted_multi_list += predicted_list.tolist()
                target_multi_list += target_list.tolist()
                video_number_list += video_number.tolist()
                image_number_list += image_number.tolist()
                loss_mse = criterion_mse(predicted_list, target_list)
                loss_all = multi_test_loss + args.lamda*loss_mse
                test_loss_1.update(multi_test_loss)
                test_loss_2.update(loss_mse)
                test_loss_all.update(loss_all)
            predicted_multi_list = [float(i) for i in predicted_multi_list]
            target_multi_list = [int(i) for i in target_multi_list]

            pd_test_loss.append(test_loss_all.avg.item())
            
            eval_res = select_keyshots(predicted_multi_list, video_number_list, image_number_list,
                                           target_multi_list, args)
            fscore_k = 0
            kendall_k = 0
            spearman_k = 0
            kendall_score = 0
            spearman_score = 0
            for i in eval_res:
                fscore_k += i[2]
                kendall_k += i[3][0]
                spearman_k += i[3][1]
                if args.dataset[:5] in ['SumMe']:
                    kendall_score+=i[4][0]
                    spearman_score+=i[4][1]
            print(len(list(args.test_dataset.split(","))))
            fscore_k /= len(list(args.test_dataset.split(",")))
            kendall_k /= len(list(args.test_dataset.split(",")))
            spearman_k /= len(list(args.test_dataset.split(",")))
            if args.dataset[:5] in ['SumMe']:
                kendall_score /= len(list(args.test_dataset.split(",")))
                spearman_score /= len(list(args.test_dataset.split(",")))
                pd_kendall_score.append(kendall_score)
                pd_spearman_score.append(spearman_score)
            pd_F_measure_k.append(fscore_k)
            pd_kendall_k.append(kendall_k)
            pd_spearman_k.append(spearman_k)
   
    if (max(pd_F_measure_k) <= fscore_k):
        save_model(model, args, fscore_k, epoch)

    print("F_kendall_k:")
    print(kendall_k)
    print("F_spearman_k:")
    print(spearman_k)
    if args.dataset in ['SumMe', 'SumMe_77','xum']:
        print("F_kendall_score:")
        print(kendall_score)
        print("F_spearman_score:")
        print(spearman_score)
    print('test_loss_1:')
    print(test_loss_1.avg.item())
    print('test_loss_2:')
    print(test_loss_2.avg.item())
    print('test_loss:')
    print(test_loss_all.avg.item())


def train(model, train_loader, optimizer,optimizer_adam, criterion,criterion_mse, epoch, args):
    global pd_lr
    global pd_loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_name = args.dataset[:5].lower()
    score_path = 'VSumMamba/datalodaers/datasets/eccv16_dataset_' + data_name + '_google_pool5.h5'
    score_file = h5py.File(score_path)
    train_loss_1 = Metric('train_loss_1')
    train_loss_2 = Metric('train_loss_2')
    train_loss = Metric('train_loss')
    model.train()
    N = len(train_loader)
    start_time = time.time()
    predicted_multi_list = []
    target_multi_list = []
    video_number_list = []
    image_number_list = []
    for batch_idx, (data, target, video_number, image_number) in enumerate(train_loader):
        lr_cur = adjust_learning_rate(args, optimizer, epoch, batch_idx, N, type=args.lr_scheduler)
        if args.cuda:
            data = data.cuda()

        predicted_list = []
        target_list = []
        optimizer.zero_grad()
        output = model(data)
        multi_target = target.permute(1, 0)
        video_number = video_number
        image_number = image_number
        multi_output = output
        multi_loss = 0
        for sequence in range(args.sequence):
            target = multi_target[sequence].cuda()
            output = multi_output[sequence]
            loss = criterion(output, target)
            multi_loss += loss

            predicted_ver2 = []
            sigmoid = nn.Sigmoid()
            outputs_sigmoid = sigmoid(output)
            for s in outputs_sigmoid:
                predicted_ver2.append(float(s[1]))
            predicted_list.append(predicted_ver2)
            target_list.append(target.tolist())

        predicted_list = torch.Tensor(predicted_list).permute(1, 0)
        predicted_list = torch.Tensor(predicted_list).reshape(args.batch_size * args.sequence)
        target_list = torch.Tensor(target_list).permute(1, 0)
        target_list = torch.Tensor(target_list).reshape(args.batch_size * args.sequence)
        video_number = video_number.reshape(args.batch_size * args.sequence)
        image_number = image_number.reshape(args.batch_size * args.sequence)

        video_number_list += video_number.tolist()
        image_number_list += image_number.tolist()
        video_number_bian = video_number.tolist()

        video_single_list = video_number_bian[::args.sequence]
        pred_last=[]
        gt_score_last=[]
        l = args.sequence
        for i in range(len(video_single_list)):
            gt_index=[]
            pred_index=[]
            index = str(video_single_list[i])
            vid = score_file['video_'+index]
            gtscore = vid['gtscore']
            image_single_video = image_number[i*l:(i+1)*l]
            for j in range(len(image_single_video)):
                if (image_single_video[j]-1)%(15)==0:
                    gt_index.append(int((image_single_video[j]-1)/(15)))
                    pred_index.append(i*l+j)
            for x in range(len(gt_index)):
                if gt_index[x]>=len(gtscore):
                    print("gt index out of range:",gt_index[x])
                gt_score_last.append(gtscore[gt_index[x]])
                if pred_index[x]>=l*args.batch_size:
                    print("index out of range")
                pred_single_score = predicted_list[pred_index[x]]
                pred_last.append(pred_single_score)

        pred_last = torch.Tensor(pred_last)
        gt_score_last = torch.Tensor(gt_score_last)

        loss_mse = criterion_mse(pred_last, gt_score_last)
        loss_mse = loss_mse.cuda()
        loss_mse = loss_mse.to(device)
        if loss_mse != loss_mse:
            print("loss_mse is Nan")
        
        loss_mse.requires_grad = True
        multi_loss /= args.sequence
        Loss_all =  multi_loss + args.lamda * loss_mse
        train_loss_1.update(multi_loss)
        train_loss_2.update(loss_mse)
        train_loss.update(Loss_all)
        Loss_all.backward()

        optimizer.step()

        # ----------------------------------------
        if (batch_idx + 1) % 100 == 0:
            memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
            used_time = time.time() - start_time
            eta = used_time / (batch_idx + 1) * (N - batch_idx)
            eta = str(datetime.timedelta(seconds=int(eta)))
            training_state = '  '.join(
                [
                    'Epoch: {}',
                    '[{} / {}]',
                    'eta: {}',
                    'lr: {:.9f}',
                    'max_mem: {:.0f}',
                    'loss_1: {:.3f}',
                    'loss_2: {:.3f}',
                    'loss: {:.3f}',
                ]
            )
            training_state = training_state.format(
                epoch + 1,
                batch_idx + 1,
                N,
                eta,
                lr_cur,
                memory,
                train_loss_1.avg.item(),
                train_loss_2.avg.item(),
                train_loss.avg.item(),
            )
            print(training_state)

        if batch_idx == N - 1:
            pd_lr.append(lr_cur)
            pd_loss.append(train_loss.avg.item())

def train_net(args):
    print("dataset:")
    print(args.dataset)
    print("Init...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args)
    print('Parameters:', sum([np.prod(p.size()) for p in model.parameters()]))
    epoch = 0
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    cudnn.benchmark = True
    if args.cuda:
        model.cuda()
    train_loader, val_loader, In_target = build_dataloader(args)
    total_target = len(train_loader) * args.batch_size * args.sequence
    A = total_target / (total_target - In_target)
    B = total_target / In_target
    
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor([A, B])).to(device)
    criterion_mse = torch.nn.MSELoss().to(device)
    optimizer_adam = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = build_optimizer(args, model)
    if args.resume:
        epoch = resume_model(model, optimizer, args)
    print("Start training...")

    global pd_epoch
    global pd_batch_size
    global pd_lr
    global pd_runtime
    global pd_loss
    global pd_F_measure_k
    global pd_kendall_k
    global pd_spearman_k
    global pd_test_loss
    global pd_kendall_score
    global pd_spearman_score
    while epoch < args.epochs:
        pd_epoch.append(epoch)
        pd_batch_size.append(args.batch_size)
        Stime = time.time()
        
        train(
                model, train_loader, optimizer,optimizer_adam,criterion,criterion_mse, epoch, args
            )
        if (epoch + 1) % args.test_epochs == 0:
            val(model, val_loader, criterion,criterion_mse, epoch, args)

        Etime = time.time()
        runtime = str(datetime.timedelta(seconds=int(Etime - Stime)))
        pd_runtime.append(runtime)
        elif args.dataset[:5] in ['SumMe']:
            ddict = {'epoch': pd_epoch,
                     'Batch_size': pd_batch_size,
                     'lr': pd_lr,
                     'runtime': pd_runtime,
                     'loss': pd_loss,
                     'F_measure_k': pd_F_measure_k,
                     'Kendall_sum': pd_kendall_k,
                     'Spearman_sum': pd_spearman_k,
                     'Kendall_score': pd_kendall_score,
                     'Spearman_score': pd_spearman_score,
                     'test_loss': pd_test_loss
                     }
        else:
            ddict = {'epoch': pd_epoch,
                     'Batch_size':pd_batch_size,
                     'lr':pd_lr,
                     'runtime':pd_runtime,
                     'loss':pd_loss,
                     'F_measure_k':pd_F_measure_k,
                     'Kendall':pd_kendall_k,
                     'Spearman':pd_spearman_k,
                     'test_loss':pd_test_loss
                     }

        dataframe = pd.DataFrame(ddict)
        
        csv_path = "save_results/csv/" + args.dataset + "/Record_" + str(args.roundtimes) + ".csv"
        dataframe.to_csv(csv_path, index=False, sep=',')

        epoch += 1


if __name__ == "__main__":

    args = parse_args()
    train_net(args)
