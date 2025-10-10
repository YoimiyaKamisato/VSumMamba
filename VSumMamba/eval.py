import os
import numpy as np
import h5py
from VSumMamba.knapsack import knapsack
from scipy.stats import spearmanr
from scipy.stats import kendalltau
from scipy.stats import rankdata
from scipy.stats import entropy

def sum_fscore(overlap_arr, true_sum_arr, oracle_sum):
    fscores = []
    for overlap, true_sum in zip(overlap_arr, true_sum_arr):
        precision = overlap / (oracle_sum + 1e-8)
        recall = overlap / (true_sum + 1e-8)
        if precision == 0 and recall == 0:
            fscore = 0
        else:
            fscore = 2 * precision * recall / (precision + recall)
        fscores.append(fscore)
    return sum(fscores) / len(fscores)
def get_oracle_summary(user_summary):
    n_user, n_frame = user_summary.shape
    oracle_summary = np.zeros(n_frame)
    overlap_arr = np.zeros(n_user)
    oracle_sum = 0
    true_sum_arr = user_summary.sum(axis=1)
    priority_idx = np.argsort(-user_summary.sum(axis=0))
    best_fscore = 0
    for idx in priority_idx:
        oracle_sum += 1
        for usr_i in range(n_user):
            overlap_arr[usr_i] += user_summary[usr_i][idx]
        cur_fscore = sum_fscore(overlap_arr, true_sum_arr, oracle_sum)
        if cur_fscore > best_fscore:
            best_fscore = cur_fscore
            oracle_summary[idx] = 1
        else:
            break

    return oracle_summary
def eval_metrics(y_pred, y_true):

    overlap = np.sum(y_pred * y_true)
    precision = overlap / (np.sum(y_pred) + 1e-8)
    recall = overlap / (np.sum(y_true) + 1e-8)
    if precision == 0 and recall == 0:
        fscore = 0
    else:
        fscore = 2 * precision * recall / (precision + recall)

    return [precision, recall, fscore]
def eval_sk(y_pred,y_true):
    s,_=spearmanr(y_pred,y_true)
    k,_ = kendalltau(rankdata(y_pred), rankdata(y_true))
    return [k,s]
def select_keyshots(predicted_list, video_number_list,image_name_list,target_list,args):
    data_path = 'VSumMamba/dataloaders/datasets'+str(args.dataset)+".h5"
    
    data_name = args.dataset[:5].lower()
    score_path = 'VSumMamba/dataloaders/datasets/eccv16_dataset_' + data_name + '_google_pool5.h5'
    data_file = h5py.File(data_path)
    score_file = h5py.File(score_path)
    predicted_single_video = []
    predicted_single_video_list = []
    target_single_video = []
    target_single_video_list = []
    video_single_list = list(set(video_number_list))
    eval_arr = []


    for i in range(len(image_name_list)):
        if image_name_list[i] == 1 and i!=0:
            predicted_single_video_list.append(predicted_single_video)
            target_single_video_list.append(target_single_video)
            predicted_single_video = []
            target_single_video = []

        predictedL = [predicted_list[i]]
        predicted_single_video += predictedL
        targetL = list(map(int, str(target_list[i])))
        target_single_video += targetL

        if i == len(image_name_list)-1:
            predicted_single_video_list.append(predicted_single_video)
            target_single_video_list.append(target_single_video)
    video_single_list_sort = sorted(video_single_list)
    True_all_video_len = 0
    for i in range(len(video_single_list_sort)):
        index = str(video_single_list_sort[i])
        video = data_file['video_' + index]
        fea_sequencelen = (len(video['feature'][:]) // args.sequence) * args.sequence
        True_all_video_len += fea_sequencelen

    for i in range(len(video_single_list_sort)):
        index = str(video_single_list_sort[i])
        video = data_file['video_' + index]
        vid = score_file['video_' + index]
        gtscore = vid['gtscore'][:]
        picks = vid['picks'][:]
       
        cps = video['change_points'][:]
        vidlen = int(cps[-1][1]) + 1
        weight = video['n_frame_per_seg'][:]
       
        fea_sequencelen = (len(video['feature'][:])//args.sequence)*args.sequence
        for ckeck_n in range(len(video_single_list_sort)):
            dif = True_all_video_len-len(predicted_list)
            if len(predicted_single_video_list[ckeck_n]) == fea_sequencelen or len(predicted_single_video_list[ckeck_n]) == fea_sequencelen-dif:
                pred_score = np.array(predicted_single_video_list[ckeck_n])
                up_rate = vidlen//len(pred_score)
          
                break
      
        pred_score = upsample(pred_score, up_rate, vidlen)
        pred_value = np.array([pred_score[cp[0]:cp[1]].mean() for cp in cps])
        _, selected = knapsack(pred_value, weight, int(0.15 * vidlen))
        selected = selected[::-1]
        key_labels = np.zeros((vidlen,))
        for i in selected:
            key_labels[cps[i][0]:cps[i][1]] = 1
        pred_summary = key_labels.tolist()
        true_summary_arr_20 = video['user_summary'][:]
        eval_res = [eval_metrics(pred_summary, true_summary_1) for true_summary_1 in true_summary_arr_20]
   
        lengths=len(picks)
        pred_pick = np.zeros(lengths)
        for i in range(lengths):
            pred_pick[i]=pred_score[picks[i]]

        eval_res = np.mean(eval_res, axis=0).tolist() if args.dataset[:5] == "TVSum" else np.max(eval_res, axis=0).tolist()
        if args.dataset[:5] == "TVSum":
            spear = spearmanr(pred_pick, gtscore)
            s0 = spear[0]
            kend = kendalltau(rankdata(pred_pick), rankdata(gtscore))
            k0 = kend[0]
            
            eval_summe_ks = [k0,s0]
            eval_res.append(eval_summe_ks)
            
        elif args.dataset[:5] == "SumMe":
            true_sum = np.mean(true_summary_arr_20,axis=0)
            spear_sum = spearmanr(pred_summary,true_sum)
            s0_sum=spear_sum[0]
            kend_sum = kendalltau(rankdata(pred_summary),rankdata(true_sum))
            k0_sum=kend_sum[0]
            eval_summe_ks_sum = [k0_sum,s0_sum]
            spear_score = spearmanr(pred_pick, gtscore)
            s0_score = spear_score[0]
            kend_score = kendalltau(rankdata(pred_pick), rankdata(gtscore))
            k0_score = kend_score[0]
            eval_summe_ks_score = [k0_score, s0_score]
            eval_res.append(eval_summe_ks_sum)
            eval_res.append(eval_summe_ks_score)
        eval_arr.append(eval_res)
    return eval_arr

def upsample(down_arr, up_rate, vidlen):
    up_arr = np.zeros(vidlen)
    for i in range(len(down_arr)):
        for j in range(up_rate):
            up_arr[i * up_rate + j] = down_arr[i]

    return up_arr
