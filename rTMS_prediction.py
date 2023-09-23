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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score, auc, roc_curve
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold
import pandas as pd
import sys
sys.path.append(r'F:\OPT\research\fMRI\utils_for_all')
from common_utils import read_singal_fmri_ts, get_fc, keep_triangle_half, sens_spec, setup_seed, t_test, heatmap,\
    vector_to_matrix, get_net_net_connect, regression_cv, classify_cv, reg_plot_scatter, get_rois_label
import os
import numpy as np
import scipy.io as sio
from scipy.stats import pearsonr, normaltest, spearmanr, ranksums
from matplotlib.pyplot import MultipleLocator
import matplotlib 
import matplotlib.ticker as mtick
from nistats import regression
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.linear_model import LinearRegression, LogisticRegression, ARDRegression, Ridge
import random

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
save_path = r'H:\PHD\learning\research\CUD\figure_UCLA'

###################### load maxico rTMS data
def get_cud_tms_data():
    data_path = r'H:\PHD\learning\research\dataset\Mexico_TMS\ROISignals'
    ccq_info = pd.read_excel(r'H:\PHD\learning\research\dataset\Mexico_TMS\clinical\5_CCQ-N.xlsx', sheet_name=0)
    vas_info = pd.read_excel(r'H:\PHD\learning\research\dataset\Mexico_TMS\clinical\12_VAS Clinico.xlsx', sheet_name=0)
    all_file = os.listdir(data_path)
    subjects = []
    sess = []
    fc_all = []
    fd_all = []
    ccq = []
    vas = []
    treatment_label = []
    
    # a = 0
    for file in all_file:
        if 'sub-015_ses-t2' in file:
            continue
        items = file.split('_')
        ID = int(items[0].split('-')[1])
        ses = items[1].split('-')[1]
        if ses == 't0' or ses == 't1' or ses == 't2':
            # a = a+1
            # if a == 34:
            #     b=1
            file_name = items[0] + '_' + items[1] + '_' + items[2] + \
                '_' + items[3] + '_' + items[4] + '_bold_Schaefer100.csv'
            one_file_name = os.path.join(data_path, file, file_name)
            ts = read_singal_fmri_ts(one_file_name)
            fc = get_fc(ts)
            fc_all.append(fc)
            subjects.append(ID)
            sess.append(ses)
            fd_file = os.path.join(r'H:\PHD\learning\research\dataset\Mexico_TMS\fmriprep', items[0], items[1], 'func',
                                   '{}_{}_{}_desc-confounds_timeseries.tsv'.format(items[0], items[1], items[2]))
            info = pd.read_csv(fd_file, sep = '\t')['framewise_displacement'].values
            fd_all.append(info)
            ccq.append(ccq_info['ccq_n'][(ccq_info['rid'] == ID) & (ccq_info['stage'] == 'T'+ses[1])].values)
            vas.append(vas_info['vas'][(ccq_info['rid'] == ID) & (ccq_info['stage'] == 'T'+ses[1])].values)
            treatment_label.append(ccq_info['group'][(ccq_info['rid'] == ID) & (ccq_info['stage'] == 'T'+ses[1])].values)

    fd_all = np.array(fd_all)[:,1:]
    subjects = np.array(subjects)
    fc_all = np.array(fc_all)
    half_fc = keep_triangle_half(
        fc_all.shape[1] * (fc_all.shape[1]-1)//2, fc_all.shape[0], fc_all)
    sess = np.array(sess)
    subjects = np.array(subjects)
    pcd_path = r'H:\PHD\learning\research\dataset\Mexico_TMS\clinical\DEMOGRAPHIC.csv'
    Table = pd.read_csv(pcd_path)
    all_name = Table['rid'].values
    sub_id = []
    for sub in subjects:
        for i in range(all_name.shape[0]):
            if sub == all_name[i]:
                sub_id.append(i)
    age_patient = Table[['q1_age', 'q1_sex']].iloc[sub_id]
    return half_fc, age_patient.values, np.mean(fd_all,-1), subjects, sess, np.array(ccq), np.array(vas), np.array(treatment_label)

def regression_tms_finetune(pretrain_weight, feature_mask, feature, target, feature2):
    loo = KFold(n_splits=5)
    loo.get_n_splits(target)
    # np.random.shuffle(vas_change)
    y_te_true = np.zeros((len(target)))
    y_te_pred = np.zeros((len(target)))
    y_te_pred2 = np.zeros((len(feature2), 5))
    weights = []
    for i, (train_index, test_index) in enumerate(loo.split(target)):
        x_tr = feature[train_index]
        x_te = feature[test_index]

        y_tr = target[train_index]
        y_te = target[test_index]
        reg = xgb.XGBRegressor(learning_rate = 0.5, importance_type = 'weight',
                                  max_depth = 5, alpha = 1, reg_lambda = 5, n_estimators = 10).fit(x_tr[:, feature_mask], y_tr)
        reg.fit(x_tr[:, feature_mask], y_tr, feature_weights=pretrain_weight)
        
        # reg = ARDRegression().fit(x_tr, y_tr)
        # reg = LinearRegression().fit(x_tr, y_tr)
        # weight = reg.coef_
        weight = reg.feature_importances_
        weight_raw = np.zeros(4950)
        weight_raw[feature_mask] = weight
        weights.append(weight_raw)
        y_te_pred[test_index] = reg.predict(x_te[:, feature_mask])
        y_te_true[test_index] = y_te.squeeze()
        y_te_pred2[:,i] = reg.predict(feature2[:, feature_mask])
    weights = np.array(weights)
    y_te_true = np.array(y_te_true)
    y_te_pred = np.array(y_te_pred)
    r2 = r2_score(y_te_true, y_te_pred)
    r, p = pearsonr(y_te_true.squeeze(), y_te_pred.squeeze())
    return r2, r, p, weights, y_te_true.squeeze(), y_te_pred.squeeze(), y_te_pred2.squeeze()


def regression_tms(feature_mask, feature, target, feature2, target2, feature3, target3):

    y_te_true_all = []
    y_te_pred_all = []
    y_te_pred2_all = []
    y_te_pred3_all = []
    weights_all = []
    idx_all = []
    r_list = []
    r_list_2 = []
    # for j in range(20):
    for j in range(10):
        print(j)
        y_te_true = np.zeros((len(target)))
        y_te_pred = np.zeros((len(target)))
        y_te_pred2 = np.zeros((len(feature2), 5))
        y_te_pred3 = np.zeros((len(feature3), 5))
        weights = []
        loo = KFold(n_splits=5, shuffle = True)
        loo.get_n_splits(target)
        idx = []
        for i, (train_index, test_index) in enumerate(loo.split(target)):
    
            x_tr = feature[train_index]
            x_te = feature[test_index]
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x_tr)
            x_te = scaler.transform(x_te)
            x_tr3 = scaler.fit_transform(feature3)
            x_tr2 = scaler.fit_transform(feature2)
            # x_tr3 = (feature3)
            # x_tr2 = (feature2)
            y_tr = target[train_index]
            y_te = target[test_index]
            
            reg = ARDRegression().fit(x_tr[:, feature_mask], y_tr)
            # reg = LinearRegression().fit(x_tr, y_tr)
            weight = reg.coef_
            # weight = reg.feature_importances_
            weight_raw = np.zeros(4950)
            weight_raw[feature_mask] = weight
            weights.append(weight_raw)
            y_te_pred[test_index] = reg.predict(x_te[:, feature_mask])
            y_te_true[test_index] = y_te.squeeze()
            y_te_pred2[:,i] = reg.predict(x_tr2[:, feature_mask])
            y_te_pred3[:,i] = reg.predict(x_tr3[:, feature_mask])
            idx.append(test_index)
        weights = np.array(weights)
        y_te_true = np.array(y_te_true)
        y_te_pred = np.array(y_te_pred)
        y_te_true_all.append(y_te_true)
        y_te_pred_all.append(y_te_pred)
        y_te_pred3_all.append(y_te_pred3)
        y_te_pred2_all.append(y_te_pred2)
        weights_all.append(weights)
        idx_all.append(idx)
        r, _ = pearsonr(y_te_true.squeeze(), y_te_pred.squeeze())
        r_2, _ = pearsonr(target2.squeeze(), y_te_pred2.mean(-1).squeeze())
        r2 = r2_score(y_te_true.squeeze(), y_te_pred.squeeze())
        r2_2 = r2_score(target2.squeeze(), y_te_pred2.mean(-1).squeeze())

        r_list.append([r, r2])
        r_list_2.append([r_2, r2_2])
    y_te_true_all = np.array(y_te_true_all)
    y_te_pred_all = np.array(y_te_pred_all)
    y_te_pred3_all = np.array(y_te_pred3_all)
    y_te_pred2_all = np.array(y_te_pred2_all)
    weights_all = np.array(weights_all)
    r2 = r2_score(y_te_true_all.mean(0), y_te_pred_all.mean(0))
    r, p = pearsonr(y_te_true_all.mean(0).squeeze(), y_te_pred_all.mean(0).squeeze())
    r2_2 = r2_score(target2.squeeze(), y_te_pred2_all.mean(-1).mean(0))
    r_2, p_2 = pearsonr(target2.squeeze(), y_te_pred2_all.mean(-1).mean(0).squeeze())
    r2_3 = r2_score(target3.squeeze(), y_te_pred3_all.mean(-1).mean(0))
    r_3, p_3 = pearsonr(target3.squeeze(), y_te_pred3_all.mean(-1).mean(0).squeeze())
    
    return r2, r, p, r2_2, r_2, p_2, r2_3, r_3, p_3, weights_all, y_te_true_all.mean(0).squeeze(),\
        y_te_pred_all.squeeze(), y_te_pred2_all.squeeze(), y_te_pred3_all.mean(-1).mean(0).squeeze(), idx_all, r_list, r_list_2


def regression_tms_permute(feature_mask, feature, target, feature2, target2, method = 'permute'):
    r2_all = []
    r_all = []
    r2_all2 = []
    r_all2 = []
    for j in range(1000):
        print(j)
        loo = KFold(n_splits=5)
        loo.get_n_splits(target)
        y_te_true = np.zeros((len(target)))
        y_te_pred = np.zeros((len(target)))
        y_te_pred2 = np.zeros((len(feature2), 5))

        for i, (train_index, test_index) in enumerate(loo.split(target)):
    
            x_tr = feature[train_index]
            x_te = feature[test_index]

            if method == 'permute':
                x_tr = x_tr[:, feature_mask]
                x_te = x_te[:, feature_mask]
                np.random.shuffle(x_tr)
            else:
                mask = random.sample(range(4950),(feature_mask).sum())
                x_tr = x_tr[:, mask]
                x_te = x_te[:, mask]
            scaler = StandardScaler()
            x_tr = scaler.fit_transform(x_tr)
            x_te = scaler.fit_transform(x_te)
            x_tr2 = scaler.fit_transform(feature2)

            y_tr = target[train_index]
            y_te = target[test_index]
            
            reg = ARDRegression().fit(x_tr, y_tr)
            weight = reg.coef_
            # weight = reg.feature_importances_
            weight_raw = np.zeros(4950)
            weight_raw[feature_mask] = weight
            y_te_pred[test_index] = reg.predict(x_te)
            y_te_pred2[:,i] = reg.predict(x_tr2[:, feature_mask])
            y_te_true[test_index] = y_te.squeeze()
        y_te_true = np.array(y_te_true)
        y_te_pred = np.array(y_te_pred)
        y_te_pred2 = np.array(y_te_pred2)
        r2 = r2_score(y_te_true, y_te_pred)
        r, p = pearsonr(y_te_true.squeeze(), y_te_pred.squeeze())
        r2_all.append(r2)
        r_all.append([r, p])
        r2 = r2_score(target2, y_te_pred2.mean(-1).squeeze())
        r, p = pearsonr(target2.squeeze(), y_te_pred2.mean(-1).squeeze())
        r2_all2.append(r2)
        r_all2.append([r, p])
    return np.array(r2_all), np.array(r_all), np.array(r2_all2), np.array(r_all2)

tms_fc, tms_pcd, tms_fd, tms_sub, tms_sess, tms_ccq, tms_vas, treatment_label = get_cud_tms_data()
tms_fc_base = tms_fc[tms_sess=='t0']
tms_pcd_base = tms_pcd[tms_sess=='t0']
tms_fd_base = tms_fd[tms_sess=='t0']
tms_sub_base = tms_sub[tms_sess=='t0']
tms_sub_t1 = tms_sub[tms_sess=='t1']
tms_sub_t2 = tms_sub[tms_sess=='t2']
tms_sub_bothtime, tms_sub_base_id, tms_sub_t1_id = np.intersect1d(tms_sub_base, tms_sub_t1, return_indices=True)
tms_sub_base_id2 = np.setdiff1d(tms_sub_base, tms_sub_bothtime, assume_unique=False)
tms_sub_t12, tms_sub_t2_id, tms_sub_t2t1_id = np.intersect1d(tms_sub_base, tms_sub_t2, return_indices=True)
tms_vas_base = tms_vas[tms_sess=='t0']
tms_ccq_base = tms_ccq[tms_sess=='t0']
tms_vas_t1 = tms_vas[tms_sess=='t1']
tms_vas_t2 = tms_vas[tms_sess=='t2']
tms_ccq_t1 = tms_ccq[tms_sess=='t1']
tms_fc_t1 = tms_fc[tms_sess=='t1']
tms_vas_change = tms_vas_base[tms_sub_base_id] - tms_vas_t1
tms_vas_ratio = (tms_vas_change+1)/(tms_vas_base[tms_sub_base_id]+1)
tms_ccq_change = tms_ccq_base[tms_sub_base_id] - tms_ccq_t1
tms_ccq_ratio = tms_ccq_change/tms_ccq_base[tms_sub_base_id]
tms_fc_base_witht1 = tms_fc_base[tms_sub_base_id]
tms_fc_change = tms_fc_base_witht1 - tms_fc_t1
treatment_base = treatment_label[tms_sess=='t0']
treatment_base_witht1 = treatment_base[tms_sub_base_id].squeeze()
tms_fc_base_witht2 = tms_fc[tms_sess=='t2']
tms_vas_changet1t2 = tms_vas_t1[tms_sub_t2t1_id] - tms_vas_t2

x = sio.loadmat(r'H:\PHD\learning\research\CUD\figure_UCLA\discovery_predictive_weights_roi_ttest2.mat')['weight']
weight_mean_final = keep_triangle_half(4950, 1, np.expand_dims(x, 0)).squeeze()

################tms prediction
# weight_mean_final = np.ones((4950))
r2_active, r_active, p_active, r2_sham, r_sham, p_sham, r2_3month, r_3month, p_3month, tms_weights, active_true, \
    active_predict, sham_predict, month3_predict, idx_, r_list, r_list_2 = regression_tms(weight_mean_final!=0,
                  tms_fc_base_witht1[treatment_base_witht1==2], tms_vas_change[treatment_base_witht1==2],
                  tms_fc_base_witht1[treatment_base_witht1==1], tms_vas_change[treatment_base_witht1==1],
                  tms_fc_base_witht2, tms_vas_changet1t2)

w, p = ranksums(np.array(r_list)[:,0], np.array(r_list_2)[:,0])
r2_all = np.r_[np.array(r_list)[:,1], np.array(r_list_2)[:,1],
                np.array(r_list)[:,0], np.array(r_list_2)[:,0]]
name1 = ['Active' for i in range(10)]
name1 = np.array(name1)
name2 = ['Sham' for i in range(10)]
name2 = np.array(name2)
names = np.r_[name1, name2]
names = np.tile(names, 2)
type1 = ['$R^{2}$' for i in range(20)]
type1 = np.array(type1)
type2 = ['r' for i in range(20)]
type2 = np.array(type2)
types = np.r_[type1, type2]
colors = ["#87CEFA", "salmon"]
sns.set(rc={'figure.figsize':(10,10)})
sns.set_style(style='white')
matplotlib.rc('font', **font2)
sns.set_palette(sns.color_palette(colors))
df = {'Treatment': names, 'Performance': r2_all, 'Metric': types}
df_ = pd.DataFrame(df)
g = sns.catplot(data=df_, x="Treatment", y="Performance", col="Metric", kind="bar", height=4, aspect=.55)
g.set(yticks=np.arange(-0.5,0.65,0.5))
g.set_axis_labels("Treatment", "Performance", weight = 'bold')
g.set_titles("{col_name}", weight = 'bold')
g.set_yticks(np.arange(-0.5, 0.65, round(0.2, 3)))
plt.savefig(os.path.join(save_path, 'performance_compare.svg'))

tms_weights_mean = np.mean(np.mean(abs(tms_weights),0),0) 
tms_weights_mask = np.mean(np.mean(tms_weights!=0,0),0) 
###########################see correlation
schf_file = r'F:\PHD\learning\project\yu_file\SchaeferParcellation_135ROIs.xlsx'
schf_info = pd.read_excel(schf_file, sheet_name=0)
node_name = schf_info.iloc[:100,1].values
node_name_array = np.ones((100, 100)).astype(object)
for i in range(100):
    for j in range(100):
        new_name = node_name[i].replace('7Networks_', '')
        new_name2 = node_name[j].replace('7Networks_', '')
        node_name_array[i,j] = new_name + '-' + new_name2

node_name_array = np.array(node_name_array, dtype=object)
idx = np.triu_indices_from(node_name_array, 1)
node_name_half = node_name_array[idx]

r_p_active = []
idx = np.where(weight_mean_final!=0)[0]
for i in idx:
    r,p = pearsonr(tms_fc_base_witht1[treatment_base_witht1==2, i], tms_vas_change[treatment_base_witht1==2].squeeze())
    r_p_active.append([r,p])
r_p_active = np.array(r_p_active)
p_fdr = multipletests(r_p_active[:,1], method = 'fdr_bh')[1]
# r_p_active[:,1] = p_fdr
node_select = node_name_half[idx]

################visual the most correlated one
font = {'family' : 'Tahoma', 'weight' : 'bold', 'size' : 30}
data = pd.DataFrame({'VAS change': tms_vas_change[treatment_base_witht1==1].squeeze(),
                      'FC': tms_fc_base_witht1[treatment_base_witht1==1, 2591].squeeze()}, 
                    index=list(range(sum(treatment_base_witht1==1))))
plt.figure(figsize=(10,10)) 
ax = plt.gca()#获取边框
sns.scatterplot(x='VAS change', y='FC', data=data, sizes = 10, color = 'salmon', ax = ax)#regplot数据点只能单色，此处为了双色显示
sns.regplot(x='VAS change', y='FC', data=data, color='salmon', ax = ax)
data = pd.DataFrame({'VAS change': tms_vas_change[treatment_base_witht1==2].squeeze(),
                      'FC': tms_fc_base_witht1[treatment_base_witht1==2, 2591].squeeze()}, 
                    index=list(range(sum(treatment_base_witht1==2))))
sns.scatterplot(x='VAS change', y='FC', data=data, sizes = 10, color = '#87CEFA', ax = ax)#regplot数据点只能单色，此处为了双色显示
sns.regplot(x='VAS change', y='FC', data=data, color='#87CEFA', ax = ax)
data = pd.DataFrame({'VAS_change': tms_vas_change.squeeze(),
                      'FC': tms_fc_base_witht1[:, 2591].squeeze(),
                      'Treatment': treatment_base_witht1.squeeze()}, 
                    index=list(range(len(treatment_base_witht1))))
g = sns.FacetGrid(data, col="Treatment", height=10, aspect = 0.5)
my_pal = {1: "salmon", 2: "#87CEFA"}
g.map_dataframe(sns.regplot, x="VAS_change", y="FC", color='grey', )
g.map_dataframe(sns.scatterplot, x="VAS_change", y="FC", hue="Treatment", palette = my_pal)
g.figure.subplots_adjust(wspace=0, hspace=0)
g.set(xlim=(-8, 13), ylim=(-0.2, 1.1), xticks=[-8, 0, 8], yticks=[0.1, 0.6, 1.1])
g.tight_layout()
min_x = -8
# max_x = 9
# min_x = -4
max_x = 13
loca_x = 6
min_y = -0.2
max_y = 1.1
loca_y = 0.4
ax.set_xticks(np.arange(min_x,max_x,loca_x))
plt.xticks(fontsize= font['size'])
ax.set_yticks(np.arange(min_y,max_y,loca_y))
plt.yticks(fontsize= font['size'])
ax.set_xlabel('Active VAS change', fontproperties=font)
ax.set_ylabel('{}'.format(node_select[411].replace('Default', 'DMN').replace('Limbic', 'LIM').replace('_', ' ')), fontproperties=font)
ax.spines['top'].set_color('none')  # 设置上‘脊梁’为红色
ax.spines['right'].set_color('none')  # 设置上‘脊梁’为无色
plt.legend(frameon=False)

