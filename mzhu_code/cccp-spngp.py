import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from learnspngp import query, build_bins
from spngp import structure
import sys



np.random.seed(58)

data = pd.read_csv('/export/homebrick/home/mzhu/mzhu_code/ccpp.csv')
# data_abnormal = pd.read_csv('/export/homebrick/home/mzhu/mzhu_code/data6dabnormal.csv')
data = pd.DataFrame(data).dropna() # miss = data.isnull().sum()/len(data)
# data_abnormal = pd.DataFrame(data_abnormal).dropna()
dmean, dstd = data.mean(), data.std()
data = (data-dmean)/dstd
# dmean1, dstd1 = data_abnormal.mean(), data_abnormal.std()
# data_abnormal = (data_abnormal-dmean1)/dstd1

# GPSPN on full data
train = data.sample(frac=0.8, random_state=58)
test  = data.drop(train.index)
# train_abnormal = data_abnormal.sample(frac=0.8, random_state=58)
# test_abnormal  = data_abnormal.drop(train_abnormal.index)
x, y = train.iloc[:, :-1].values, train.iloc[:, -1].values.reshape(-1,1)
print(data)
opts = {
    'min_samples':          0,
    'X':                    x, 
    'qd':                   4, 
    'max_depth':            4,
    'max_samples':      10**10,
    'log':               True,
    'jump':              True,
    'reduce_branching':  True
}
root_region, gps_ = build_bins(**opts)
#root_region, gps_ = build(X=x, delta_divisor=3, max_depth=2)
root, gps, gps1, gps2        = structure(root_region, gp_types=['rbf'])  #modified



for i, gp in enumerate(gps):
    idx = query(x, gp.mins, gp.maxs)
    gp.x, gp.y = x[idx], y[idx]

    print(f"Training GP set1 {i+1}/{len(gps)} ({len(idx)})") #modified
    gp.init()

##modified
for i, gp in enumerate(gps1):
    idx = query(x, gp.mins, gp.maxs)
    gp.x, gp.y = x[idx], y[idx]

    print(f"Training GP set2 {i+1}/{len(gps)} ({len(idx)})") #modified
    gp.init1(cuda=True)

for i, gp in enumerate(gps2):
    idx = query(x, gp.mins, gp.maxs)
    gp.x, gp.y = x[idx], y[idx]

    print(f"Training GP set3 {i+1}/{len(gps)} ({len(idx)})") #modified
    gp.init2(cuda=True)

root.update()


mll_=[]
mll_abnormal_=[]
cov = []
cov_abnormal =[]
RMSE_=[]
RMSE_abnormal_=[]
all_mll=[]
all_mll_abnormal=[]
all_rmse=[]
all_rmse_abnormal=[]
for smudge in np.arange(0, 0.5, 0.05):
    mu_s, cov_s, mll = root.forward(test.iloc[:, :-1].values,test.iloc[:,-1].values, smudge=smudge)
    # mu_s_abnormal, cov_s_abnormal, mll_abnormal = root.forward(test_abnormal.iloc[:, :-1].values, test_abnormal.iloc[:, -1].values, smudge=smudge)

    mu_s = (mu_s.ravel() * dstd.iloc[-1]) + dmean.iloc[-1]

    # mu_s_abnormal = (mu_s_abnormal.ravel() * dstd1.iloc[-1]) + dmean1.iloc[-1]

    mu_t = (test.iloc[:, -1]*dstd.iloc[-1]) + dmean.iloc[-1]
    # mu_t_abnormal = (test_abnormal.iloc[:, -1] * dstd1.iloc[-1]) + dmean1.iloc[-1]
    sqe = (mu_s - mu_t.values)**2



    rmse = np.sqrt(sqe.sum()/len(test))
    # sqe_abnormal = (mu_s_abnormal - mu_t_abnormal.values) ** 2

    # rmse_abnormal = np.sqrt(sqe_abnormal.sum() / len(test))

    mll_.append(np.mean(mll))

    # mll_abnormal_.append(np.mean(mll_abnormal))


    cov.append(np.mean(cov_s))
    # cov_abnormal.append(np.mean(cov_s_abnormal))
    RMSE_.append(rmse)
    # RMSE_abnormal_.append(rmse_abnormal)
    if smudge == 0:
        all_rmse =np.sqrt(sqe)
        # all_rmse_abnormal=np.sqrt(sqe_abnormal)
        all_mll.extend(mll)
        # all_mll.extend(mll_abnormal)
    # print("mean_mll=", np.mean(mll))
    # print("mean_cov=", np.mean(cov_s))
    # print(f"SPN-GP (smudge={round(smudge, 4)}) \t RMSE: {rmse}")
    # print("mean_mll_abnormal=", np.mean(mll_abnormal))
    # print("mean_cov_abnormal=", np.mean(cov_s_abnormal))
    # print(f"SPN-GP (smudge={round(smudge, 4)}) \t RMSE_abnormal: {rmse_abnormal}")

# print('mse_normal', all_rmse)
# print('mse_abnoraml',all_rmse_abnormal)
# print('mll_normal', all_mll)
# print('mll_abnormal', all_mll_abnormal)
print('mll_normal:', mll_)
print('mll_abnormal:', mll_abnormal_)
print('RMSE_normal:', RMSE_)
print('RMSE_abnormal:', RMSE_abnormal_)
# print('all_normal_mse',all_rmse)
# print('all_abnormal_mse',all_rmse_abnormal)
# print('all_normal_mll', all_mll)
# print('all_abnormal_mll',all_mll_abnormal)
y1 =np.hstack(all_mll)
all_rmse_improved = np.concatenate((all_rmse, all_rmse_abnormal))
# # print('all_mll',y1)
#
# print('mll_normal:', mll_)
# print('mll_abnormal:', mll_abnormal_)
# print('RMSE_normal:', RMSE_)
# print('RMSE_abnormal:', RMSE_abnormal_)
#
np.savetxt('all_rmse_improved_iqr.csv', [all_rmse_improved], delimiter=',')
#
np.savetxt('all_mll_improved_iqr.csv', y1, delimiter=',')
# fpr, tpr, thresholds  =  roc_curve(y_test, scores)
# roc_auc = auc(fpr,tpr)
# plt.figure(figsize=(6,6))
# plt.title('Validation ROC')
# plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
