import subprocess
subprocess.call(["bash","apt_install.sh"])

import os
import numpy as np
import gc
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score
np.random.seed(42)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from constants import model_params, base_data_path, measure_performance_only
from utils import Discriminator_Model


try:
    import foundations
    for k,v in model_params.items():
        foundations.log_param(k, v)
except Exception as e:
    print(e)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


print(f"loading the preprocessed training, validation and test data.")
train_data = []
val_data = []
test_data = []
for i in range(len(base_data_path)):
    train_data.extend(np.load(os.path.join(f'{base_data_path[i]}/preprocessed_data', f'train_preproc_aug.npy'), allow_pickle=True))
    val_data.extend(np.load(os.path.join(f'{base_data_path[i]}/preprocessed_data', f'val_preproc.npy'), allow_pickle=True))
    test_data.extend(np.load(os.path.join(f'{base_data_path[i]}/preprocessed_data', f'test_preproc.npy'), allow_pickle=True))

print("sorting the train_data, val_data and test data")
# Sort train data by the sequence length
train_data = sorted(train_data, key = lambda x: len(x[0]))
val_data = sorted(val_data, key = lambda x: len(x[0]))
test_data = sorted(test_data, key = lambda x: len(x[0]))


xtrain = [x[0] for x in train_data]
ytrain = [x[1] for x in train_data]
xval = [x[0] for x in val_data]
yval = [x[1] for x in val_data]
xtest = [x[0] for x in test_data]
ytest = [x[1] for x in test_data]

del train_data, val_data, test_data
gc.collect()

print("calculating class balance of train set")
ytrain_freq = stats.itemfreq(ytrain)
print(ytrain_freq)

print("calculating class balance of test set")
ytest_freq = stats.itemfreq(ytest)
print(ytest_freq)


print("Initializing model")
if measure_performance_only:
    pretrained_model_name = 'saved_model_240_8_32_0.05_1_50_0_0.0001_500_156_2_True_True_fitted_objects.h5'
    model_class = Discriminator_Model(load_pretrained=True, saved_model_name=pretrained_model_name, real_test_mode=True)
else:
    model_class = Discriminator_Model(load_pretrained=False, real_test_mode=True)
    print("training model")
    model_class.train(xtrain, ytrain, xval, yval)
print("optimizing threshold probability")
model_class.optimize_threshold(xtrain, ytrain, xval, yval)
print(f"optimum threshold is {model_class.opt_threshold}")
print("evaluating trained model")
model_class.evaluate(xtrain, ytrain, xval, yval)
print("calculating test performance")
y_test_pred_labels = model_class.predict_labels(xtest, threshold=model_class.opt_threshold)
test_acc = accuracy_score(ytest, y_test_pred_labels)
test_f1_score = f1_score(ytest, y_test_pred_labels)
print(f"Test set accuracy: {test_acc}, f1_score: {test_f1_score} ")

print("calculating performance on real test set")
real_test_acc_score, real_test_f1_score_val = model_class.inference_on_real_data(threshold=model_class.opt_threshold)
print(f"Realtalk test set accuracy: {real_test_acc_score}, f1_score: {real_test_f1_score_val} ")

try:
    foundations.log_metric('test_accuracy', np.round(test_acc, 2))
    foundations.log_metric('test_f1_score', np.round(test_f1_score, 2))
    foundations.log_metric('realtalk_accuracy', np.round(real_test_acc_score, 2))
    foundations.log_metric('realtalk_f1_score', np.round(real_test_f1_score_val, 2))

except:
    print("foundations command not found")
