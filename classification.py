# -*- coding: utf-8 -*-
"""
Created on Wed Apr  5 10:27:50 2023

@author: 99488
"""

import networkx
import copy
import matplotlib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy import stats
from neuroCombat import neuroCombat
from statsmodels.stats.multitest import multipletests
# import shap
from sklearn.inspection import permutation_importance
from collections import namedtuple, OrderedDict
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, auc, roc_curve
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut
import pandas as pd
import sys
sys.path.append(r'F:\OPT\research\fMRI\utils_for_all')
from common_utils import read_singal_fmri_ts, get_fc, keep_triangle_half, sens_spec, setup_seed, t_test, heatmap,\
    vector_to_matrix, get_net_net_connect, regression_cv, classify_cv, reg_plot_scatter, get_rois_label
import os
import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr, normaltest, spearmanr, ranksums, chi2_contingency
from matplotlib.pyplot import MultipleLocator
import matplotlib 
import matplotlib.ticker as mtick
from nistats import regression
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LinearRegression, LogisticRegression, ARDRegression, Ridge, BayesianRidge, MultiTaskElasticNet, MultiTaskLasso
import random
from sklearn.svm import SVR
import xgboost as xgb
import mrmr

net_info_shf_100 = {'VIS': ([0,9],[50,58]), 'SMN': ([9,15],[58,66]), 'DAN': ([15,23],[66,73]), 'VAN': ([23,30],[73,78]), 
                                                'LIM': ([30,33],[78,80]), 'FPC': ([33,37],[80,89]), 'DMN':([37,50],[89,100])}

font2 = {'family': 'Tahoma', 'weight': 'bold', 'size': 30}
matplotlib.rc('font', **font2)
setup_seed(6)

Color = namedtuple('RGB', 'red, green, blue')
colors = {}  # dict of colors

class RGB(Color):
    def hex_format(self):
        return '#{:02X}{:02X}{:02X}'.format(self.red, self.green, self.blue)

# Color Contants
type1 = RGB(1, 0.702, 0.702)
type2 = RGB(0.651, 0.8706, 0.9647)
# type3 = RGB(0.49, 0.686, 0.223)
type3 = RGB(0.612, 0.894, 0.491)
type4 = RGB(0.612, 0.894, 0.491)

folds = 10
p_thre = 0.05 #0.01
save_path = r'H:\PHD\learning\research\CUD\figure_UCLA'

# sio.savemat(r'H:\PHD\learning\research\CUD\data\data_info.mat', {'discovery_fc_aug': half_fc_all, 'discovery_fc_raw': half_fc, 'discovery_label_raw':dx,
#                                                 'discovery_label_aug':dx_all, 'discovery_subID_aug':subjects_all, 'discovery_subID_raw':subjects,
#                                                 'independent_fc': half_fc_indep_raw, 'independent_label': label_indep})
data_info = sio.loadmat(r'H:\PHD\learning\research\CUD\data\data_info.mat')
half_fc_all = data_info['discovery_fc_aug']
half_fc = data_info['discovery_fc_raw']
dx = data_info['discovery_label_raw'].squeeze()
dx_all = data_info['discovery_label_aug'].squeeze()
subjects_all = data_info['discovery_subID_aug'].squeeze()
subjects = data_info['discovery_subID_raw'].squeeze()
half_fc_indep_raw = data_info['independent_fc']
label_indep = data_info['independent_label'].squeeze()
##################################
kf = StratifiedKFold(n_splits=folds, random_state=6, shuffle=True)
perform = []
weights = []
weights_inloop = []
weights_permute = []
weights_inloop_permute = []
weights_shap = []
weights_inloop_shap = []

t_p_vector_all = []
t_p_vector_all_inloop = []
acc_inloop = 0
nyu_pred_all = []
nyu_pred_all_inloop = []
pred_all_confidence = np.ones((1,3))+3

for train_index, test_index in kf.split(half_fc, dx):
    tr_all_id = []
    for sub in subjects[train_index]:
        tr_all_id.extend(list(np.where(subjects_all==sub)[0]))
    tr_all_id = np.array(tr_all_id)
    # np.random.shuffle(tr_all_id)
    te_all_id = []
    for sub in subjects[test_index]:
        te_all_id.extend(list(np.where(subjects_all==sub)[0]))
    te_all_id = np.array(te_all_id)
    subjects_all_te = subjects_all[te_all_id]
    te_sub_unqiu, test_all_id, te_unique_id = np.unique(subjects_all_te, return_index = True, return_inverse=True)

    X_train = half_fc_all[tr_all_id]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(half_fc_all[te_all_id])
    # X_test = half_fc_all[te_all_id]
    half_fc_indep = half_fc_indep_raw
    # scaler = StandardScaler()

    # half_fc_indep = scaler.transform(half_fc_indep)
    # half_fc_indep[:half_fc_hc.shape[0]] = scaler.fit_transform(half_fc_indep[:half_fc_hc.shape[0]])
    # half_fc_indep[:half_fc_patient_raw.shape[0]] = scaler.fit_transform(half_fc_indep[:half_fc_patient_raw.shape[0]])
    # half_fc_indep[-half_fc_patient_raw.shape[0]:] = scaler.fit_transform(half_fc_indep[-half_fc_patient_raw.shape[0]:])
    # half_fc_indep = half_fc_indep_raw
    # half_fc_patient = half_fc_patient_raw
    y_train = dx_all[tr_all_id]
    y_test = dx_all[te_all_id]
    y_test_uni = dx_all[te_all_id][test_all_id]

    t_p_vector = np.zeros((4950, 2))
    for i in range(4950):
        t_p_vector[i, 0],  t_p_vector[i, 1] = t_test(X_train[y_train==0, i], X_train[y_train==1, i], 0.05, False)
        # t_p_vector[i, 0],  t_p_vector[i, 1] = stats.pointbiserialr(X_train[:, i], y_train)
        # stats.pointbiserialr(a, b)

    t_p_vector_all.append(t_p_vector)
    X_train = X_train[:,t_p_vector[:,1]<=p_thre]
    X_test = X_test[:,t_p_vector[:,1]<=p_thre]
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    # sample_weights = compute_sample_weight(class_weight={0: 1, 1: 3}, y=y_train)
    
    clf = xgb.XGBClassifier(learning_rate = 0.5, importance_type = 'weight',
                              max_depth = 5, alpha = 10, reg_lambda = 15, n_estimators = 10).fit(X_train, y_train, sample_weight=sample_weights)
    weight = clf.feature_importances_

    weights.append(weight.squeeze())


    pred_test_all = clf.predict(X_test)
    nyu_pred = clf.predict(half_fc_indep[:,t_p_vector[:,1]<=p_thre])
    nyu_pred_all.append(nyu_pred)
    
    pred_test_uni = []
    predPROB_test_uni = []
    for te in te_sub_unqiu:
        y_pred = pred_test_all[np.where(subjects_all_te==te)[0]]
        y_predPROB = clf.predict_proba(X_test)[np.where(subjects_all_te==te)[0]]
        if len(set(y_pred))==1:
            pred_test_uni.append(y_pred[0])
            predPROB_test_uni.append(y_predPROB[0])
        else:
            if len(y_pred)==3 or len(y_pred)==5:
                pred_test_uni.append(np.argmax(np.bincount(y_pred)))
                predPROB_test_uni.append(np.mean(y_predPROB[np.argmax(np.bincount(y_pred))==y_pred], 0))
            # elif len(y_pred)==2:
            else:
                predict_te_inloop = []
                prePROB_te_inloop = []
                for k in range(5):
                    # np.random.shuffle(tr_all_id)
                    # tr_all_id = tr_all_id[:int(len(tr_all_id)*0.9)]
                    mask = subjects_all!=te
                    idx_ = np.random.choice(sum(mask), sum(mask))
                    X_train_inloop_raw = half_fc_all[mask][idx_]
                    scaler = StandardScaler()
                    X_train_inloop = scaler.fit_transform(X_train_inloop_raw)
                    y_train_inloop = dx_all[mask][idx_]
                    # X_train_inloop = X_train_inloop_raw
                    y_train_inloop = dx_all[mask][idx_]
                    t_p_vector = np.zeros((4950, 2))
                    for i in range(4950):
                        t_p_vector[i, 0],  t_p_vector[i, 1] = t_test(X_train_inloop[y_train_inloop==0, i], X_train_inloop[y_train_inloop==1, i], 0.05, False)
                        # t_p_vector[i, 0],  t_p_vector[i, 1] = stats.pointbiserialr(X_train_inloop[:, i], y_train_inloop)

                    t_p_vector_all_inloop.append(t_p_vector)
                    X_train_inloop = X_train_inloop[:,t_p_vector[:,1]<=p_thre]
                    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train_inloop)
                    # sample_weights = compute_sample_weight(class_weight={0: 1, 1: 3}, y=y_train_inloop)

                    clf_inloop = xgb.XGBClassifier(learning_rate = 0.5, importance_type = 'weight',
                                              max_depth = 5, alpha = 10, reg_lambda = 15, n_estimators = 10).fit(X_train_inloop, y_train_inloop, sample_weight=sample_weights)
                    weight = clf_inloop.feature_importances_

                    weights_inloop.append(weight)

                    pred_test_inloop = clf_inloop.predict(half_fc_all[~mask][:,t_p_vector[:,1]<=p_thre])
                    predPROB_test_inloop = clf_inloop.predict_proba(half_fc_all[~mask][:,t_p_vector[:,1]<=p_thre])
                    predict_te_inloop.extend(list(pred_test_inloop))
                    prePROB_te_inloop.extend(list(predPROB_test_inloop))
                    nyu_pred_all_inloop.append(clf_inloop.predict(half_fc_indep[:,t_p_vector[:,1]<=p_thre]))

                    pred_all_confidence = np.r_[pred_all_confidence, np.c_[clf_inloop.predict_proba(half_fc_all[~mask][:,t_p_vector[:,1]<=p_thre]),
                                                                                                    dx_all[~mask]]]

                        
                pred_test_uni.append(np.argmax(np.bincount(predict_te_inloop)))
                predPROB_test_uni.append(np.mean(np.array(prePROB_te_inloop)\
                                                  [np.argmax(np.bincount(predict_te_inloop))==predict_te_inloop], 0))
                acc = np.argmax(np.bincount(predict_te_inloop))==dx[subjects==te]
                acc_inloop = acc_inloop + acc
    pred_tr = np.expand_dims(clf.predict(X_train), -1)
    sen, spe = sens_spec(pred_test_uni, y_test_uni)
    acc = accuracy_score(y_test_uni,pred_test_uni)
    fpr, tpr, thresholds = roc_curve(y_test_uni,np.array(predPROB_test_uni)[:,1])
    AUC = auc(fpr, tpr)
    print('training acc: {}'.format(accuracy_score(y_train,pred_tr)))
    perform.append([acc, sen, spe, AUC])

nyu_pred_all = np.array(nyu_pred_all)
nyu_pred_all_inloop = np.array(nyu_pred_all_inloop)
nyu_pred_all_inloop_mean = np.mean(nyu_pred_all_inloop, 0)>0.5
nyu_pred_all_mean = ((np.mean(nyu_pred_all, 0) + nyu_pred_all_inloop_mean/5*(6/130))>0.5)*1
# label_indep
sen_ind, spe_ind = sens_spec(nyu_pred_all_mean, label_indep)
acc_ind = accuracy_score(label_indep,nyu_pred_all_mean)

# ##############sex perform
# perf_site = []
# for i in range(1000):
#     # np.random.shuffle(label_indep)
#     idx_ = np.random.choice(np.arange(half_fc_hc.shape[0]+half_fc_patient_raw.shape[0]), half_fc_hc.shape[0])
#     nyu_pred_all_mean_ = np.r_[nyu_pred_all_mean[idx_], nyu_pred_all_mean[(half_fc_hc.shape[0]+half_fc_patient_raw.shape[0]):]]
#     label_indep_ = np.r_[label_indep[idx_], label_indep[(half_fc_hc.shape[0]+half_fc_patient_raw.shape[0]):]]
#     sen_ind, spe_ind = sens_spec(nyu_pred_all_mean_, label_indep_)
#     acc_ind = accuracy_score(label_indep_,nyu_pred_all_mean_)
#     perf_site.append(np.r_[acc_ind, sen_ind, spe_ind].squeeze())

# perf_site = np.array(perf_site)
# ######################
# perform = perf_site
perform = np.array(perform)
plt.figure(figsize=(13,13))
ax = plt.gca()
colors = ["#87CEFA", "salmon", type3]
sns.set_palette(sns.color_palette(colors))
name1 = ['Accuracy' for i in range(10)]
name1 = np.array(name1)
name2 = ['Sensitivity' for i in range(10)]
name2 = np.array(name2)
name3 = ['Specificity' for i in range(10)]
name3 = np.array(name3)
names = np.r_[name1, name2, name3]
perf = np.r_[perform[:,0], perform[:,1], perform[:,2]].squeeze()
df = {'Performance': names, 'Score': perf}
df_ = pd.DataFrame(df)
sns.barplot(x='Performance', y='Score', data=df_, ax = ax)

# plt.axhline(0, c = "grey", ls = '-')

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

change_width(ax, .4)
ax.tick_params(top=False, bottom=False,
                labeltop=False, labelbottom=True)
ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
ax.set_ylabel('Performance', fontproperties=font2)
plt.ylim(0, 0.8)  

# # save_file = os.path.join(save_path, 'perfprmance_aug_threetime_0.82_0.80_0.85_witherror.svg')
# save_file = os.path.join(save_path, 'perfprmance_site_classification_subsample1000.svg')
# if save_file:
#     plt.savefig(save_file, format = 'svg', bbox_inches = 'tight')

plt.figure(figsize=(13,13))
ax = plt.gca()
colors = ["#87CEFA", "salmon", type3]
sns.set_palette(sns.color_palette(colors))
name1 = ['Accuracy']
name1 = np.array(name1)
name2 = ['Sensitivity']
name2 = np.array(name2)
name3 = ['Specificity']
name3 = np.array(name3)
names = np.r_[name1, name2, name3]
perf = np.r_[acc_ind, sen_ind, spe_ind].squeeze()
df = {'Score': names, 'Performance': perf}
df_ = pd.DataFrame(df)
sns.barplot(x='Score', y='Performance', data=df_, ax = ax, errwidth=0)
change_width(ax, .4)
ax.tick_params(top=False, bottom=False,
                labeltop=False, labelbottom=True)
ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
ax.set_ylabel('Performance', fontproperties=font2)
plt.ylim(0, 0.8)  

# save_file = os.path.join(save_path, 'performance_reproduce_site_NUY_TMS_UCLAall.svg')
# if save_file:
#     plt.savefig(save_file, format = 'svg', bbox_inches = 'tight')

########################pattern vis
cmap = 'Purples'
roiORnet = 'roi'
predict_weight = np.zeros((4950,10))
predict_weight_permute = np.zeros((4950,10))
predict_weight_shap = np.zeros((4950,10))
predict_weight_inloop = np.zeros((4950,len(weights_inloop)))
predict_weight_permute_inloop = np.zeros((4950,len(weights_inloop)))
predict_weight_shap_inloop = np.zeros((4950,len(weights_inloop)))

for i in range(10):
    mask_id = t_p_vector_all[i][:,1]<=p_thre
    predict_weight[mask_id,i] = weights[i]
weight_mean = np.mean(predict_weight, -1)

for i in range(len(weights_inloop)):
    mask_id = t_p_vector_all_inloop[i][:,1]<=p_thre
    predict_weight_inloop[mask_id,i] = weights_inloop[i]

weight_mean_inloop = np.mean(predict_weight_inloop, -1)

# x = sio.loadmat(r'H:\PHD\learning\research\CUD\figure_UCLA\'discovery_predictive_weights_{}_ttest.mat'.format(roiORnet))['weight']
# weight_mean_final = keep_triangle_half(4950, 1, np.expand_dims(x, 0)).squeeze()
weight_mean_final = weight_mean+weight_mean_inloop/(len(weight_mean_inloop)/5)
label_idx = np.arange(0,100,1)
t_p_vector = np.zeros((4950, 2))
for i in range(4950):
    t_p_vector[i, 0],  t_p_vector[i, 1] = t_test(half_fc_all[dx_all==0, i], half_fc_all[dx_all==1, i], 0.05, False)
p = t_p_vector[weight_mean_final!=0, 1]
t_p_vector[weight_mean_final!=0, 1]
p_fdr = multipletests(p, method = 'fdr_bh')[1]
t_p_vector[weight_mean_final!=0, 1] = p_fdr
t_p_vector[t_p_vector[:,1]>0.05,:] = 0
t_p_vector[weight_mean_final==0,:] = 0
t_vector_mean_sys, _ = vector_to_matrix(t_p_vector[:,0])
p_vector_mean_sys, _ = vector_to_matrix(weight_mean_final)
cbarlabel = 't-value'
plt.figure(figsize =(15,15))
ax = plt.gca()
im, cbar = heatmap(t_vector_mean_sys.squeeze(), label_idx, ax=ax, cmap='RdBu_r', connect_type=roiORnet,
                    cbarlabel=cbarlabel, half_or_full = 'half', with_diag = True)
save_name = 'discovery_predictive_weights_{}_ttest.svg'.format(roiORnet)
# plt.savefig(os.path.join(save_path, save_name), bbox_inches = 'tight')
sio.savemat(os.path.join(save_path, 'discovery_predictive_weights_{}_ttest.mat'.format(roiORnet)),{'weight': t_vector_mean_sys})

#############pattern similarity
# from scipy.spatial import distance
# similarity = 1 - distance.cosine(t_p_vector_[weight_mean_final!=0,0], t_p_vector[weight_mean_final!=0,0])
# similarity2 = pearsonr(t_p_vector_[weight_mean_final!=0,0], t_p_vector[weight_mean_final!=0,0])
# simlis = []
# for i in range(1000):
#     idx_ = np.random.choice(t_p_vector_.shape[0], sum(weight_mean_final!=0))
#     similarity_ = pearsonr(t_p_vector_[idx_,0], t_p_vector[idx_,0])
#     simlis.append(similarity_)
# simlis = np.array(simlis)
# ############permute
# p = (sum(simlis[:,0]>0.77)+1)/1001
# plt.figure(figsize =(10,10))
# ax = plt.gca()
# sns.kdeplot(simlis[:,0], shade=True, bw_adjust=.1, alpha=.5, linewidth=0, color = '#87CEFA')
# plt.axvline(x=0.77, color='black', linestyle='--')
# ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
# ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
# ax.spines['left'].set_color('none')  # 设置上‘脊梁’为无色
# ax.axes.get_yaxis().set_visible(False)
# ax.tick_params(axis='x', which='major', labelsize=50)
# ax.set_xticks(np.arange(0.1, 0.8, 0.3))
# # plt.savefig(os.path.join(save_path, 'r_active_subsample_true.svg'), format = 'svg', bbox_inches = 'tight')
