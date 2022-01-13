"""
   The trained model is used here to obtain test results.
"""
import warnings
warnings.filterwarnings('ignore')
from moment_update import moment_update_ema
from sklearn.model_selection import KFold
from braindecode.torch_ext.util import np_to_var
import torch.nn as nn
import torch.optim as optim
import time
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
from tqdm import trange
from until.deep4_net import Deep4Net_base
from until.eegnet import eegnet
from until.ConvNet import ShallowFBCSPNet
import until.lr_schedule as lr_schedule
from until.constraints import MaxNormDefaultConstraint
from until.constraints_EEGNet import MaxNormDefaultConstraint_EEGNet
import losses_supcon
import matplotlib.pyplot as plt
import argparse
from saveResult import write_log
from until.loadData import *

parser = argparse.ArgumentParser(description='input the dataset dir path.')

parser.add_argument('--backbone_net_all', type=str, default=['shallow'], help='choose model: shallow, deep, eegnet')
parser.add_argument('--test_interval', type=int, default=1, help='iter')
parser.add_argument('--max_epochs', type=int, default=1500)
parser.add_argument('--folds', type=int, default=10, help='x fold cross validation')
parser.add_argument('--subject', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9])
parser.add_argument('--log_path', type=str, default='./result/', help='folder')
parser.add_argument('--foldname', type=str, default='result.txt', help='filename')
parser.add_argument('--alpha1', type=float, default=1, help='parameter of co-contrastive')
parser.add_argument('--alpha2', type=float, default=0.001, help='parameter of mse_loss')
parser.add_argument('--batchsize', type=int, default=120)
parser.add_argument('--dataset', type=str, default="2a", help='choose: 2a, 2b')

args = parser.parse_args()

optim_dict = {"Adam": optim.Adam, "SGD": optim.SGD}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True

torch.cuda.set_device(0)

acc = list()
kappa_value = list()
recall_value = list()
preci_value = list()
F1_value = list()

write_log(args.max_epochs, args)
write_log(args.subject, args)
ACC = list()
kappa_all = list()
recall_all = list()
preci_all = list()
F1_all = list()
time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

for backbone_net in args.backbone_net_all:
    for subject_id in args.subject:
        # load original data(preprocessed)
        ori_folder_tr = './data/' + args.dataset + '/original/traindata.mat'
        ori_folder_te = './data/' + args.dataset + '/original/testdata.mat'
        trainingdataall_1, traininglabelall_1 = originalData(ori_folder_tr, ori_folder_te, subject_id)
        # load A'' data
        mixcat_folder_tr = './data/' + args.dataset + '/mixcat/traindata.mat'
        mixcat_folder_te = './data/' + args.dataset + '/mixcat/testdata.mat'
        trainingdataall_2, traininglabelall_2 = mixcatData(mixcat_folder_tr, mixcat_folder_te, subject_id)
        # load A data
        cutmix_folder_tr = './data/' + args.dataset + '/timecutmix/traindata.mat'
        cutmix_folder_te = './data/' + args.dataset + '/timecutmix/testdata.mat'
        trainingdataall_3, traininglabelall_3 = cutmixData(cutmix_folder_tr, cutmix_folder_te, subject_id)
        # process label
        traininglabelall_1, traininglabelall_2, traininglabelall_3 = processLabel(traininglabelall_1, traininglabelall_2, traininglabelall_3)

        kf = KFold(n_splits=args.folds, shuffle=True, random_state=20210825)
        fold = 0
        for train_index, test_index in kf.split(trainingdataall_1):
            fold = fold + 1
            print("ten folds cross-validation: S=" + str(subject_id) + ", fold=" + str(fold))

            # test data
            testdata = trainingdataall_1[test_index]
            testlabel = traininglabelall_1[test_index]

            X_test = Variable(testdata)
            X_test = torch.unsqueeze(X_test, dim=3)
            testinglabel = Variable(testlabel)

            # set base network
            classes = torch.unique(testlabel, sorted=False).numpy()
            n_classes = len(classes)
            n_chans = int(testdata.shape[1])
            input_time_length = testdata.shape[2]

            if backbone_net == 'shallow':
                model = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                                        final_conv_length='auto')
            elif backbone_net == 'deep':
                model = Deep4Net_base(n_chans, n_classes, input_time_length=input_time_length,
                                      final_conv_length='auto')
            elif backbone_net == 'eegnet':
                kernel_length = 125
                model = eegnet(n_classes=n_classes, in_chans=n_chans, input_time_length=input_time_length,
                               kernel_length=kernel_length)

            if cuda:
                model.cuda()

            if backbone_net == 'eegnet':
                model_constraint = MaxNormDefaultConstraint_EEGNet()
            else:
                model_constraint = MaxNormDefaultConstraint()

            # metrics
            test_accuracy = np.zeros(shape=[0], dtype=float)
            test_recall = np.zeros(shape=[0], dtype=float)
            test_f1 = np.zeros(shape=[0], dtype=float)
            test_preci = np.zeros(shape=[0], dtype=float)
            test_kappa = np.zeros(shape=[0], dtype=float)

            model.eval()

            # test in the train
            b_x = X_test.float()
            b_y = testinglabel.long()
            # load pre-training model parameters
            path = './model_parameters/' + str(args.dataset) + '/' + str(args.dataset) + '_S' +\
                   str(subject_id) + '_F' + str(fold) + '.pth'
            model.load_state_dict(torch.load(path))

            with torch.no_grad():
                input_vars = np_to_var(b_x, pin_memory=False).float()
                labels_vars_test = np_to_var(b_y, pin_memory=False).type(torch.LongTensor)
                input_vars = input_vars.cuda()
                labels_vars_test = labels_vars_test.cuda()

                outputs = model(input_vars)

                y_test_pred = torch.max(outputs, 1)[1].cpu().data.numpy().squeeze()

                acc = metrics.accuracy_score(labels_vars_test.cpu().data.numpy(), y_test_pred)
                recall = metrics.recall_score(labels_vars_test.cpu().data.numpy(), y_test_pred,
                                              average='macro')
                f1 = metrics.f1_score(labels_vars_test.cpu().data.numpy(), y_test_pred, average='macro')
                preci = metrics.precision_score(labels_vars_test.cpu().data.numpy(), y_test_pred,
                                                average='macro')
                kappa = metrics.cohen_kappa_score(labels_vars_test.cpu().data.numpy(), y_test_pred)

                test_accuracy = np.append(test_accuracy, acc)
                test_recall = np.append(test_recall, recall)
                test_f1 = np.append(test_f1, f1)
                test_preci = np.append(test_preci, preci)
                test_kappa = np.append(test_kappa, kappa)

        # save ten-folds result
        acc_result = test_accuracy[-10:]
        recall_result = test_recall[-10:]
        f1_result = test_f1[-10:]
        preci_result = test_preci[-10:]
        kap_result = test_kappa[-10:]

        print("S=" + str(subject_id))
        print("all 10 folds ACC:", acc_result)
        s = "kf: 1-10"
        write_log("S = " + str(subject_id), args)
        write_log(s, args)
        write_log("ACC", args)
        write_log(list(acc_result), args)

        write_log("recall", args)
        write_log(list(recall_result), args)

        write_log("F1", args)
        write_log(list(f1_result), args)

        write_log("Precision", args)
        write_log(list(preci_result), args)

        write_log("Kappa", args)
        write_log(list(kap_result), args)

        ACC.append(acc_result.mean())
        kappa_all.append(kap_result.mean())
        F1_all.append(f1_result.mean())
        preci_all.append(preci_result.mean())
        recall_all.append(recall_result.mean())

        acc = list()
        kappa_value = list()
        recall_value = list()
        F1_value = list()
        preci_value = list()

write_log("all subject test  result：", args)
write_log("ACC", args)
write_log(ACC, args)
write_log("kappa", args)
write_log(kappa_all, args)
write_log("recall", args)
write_log(recall_all, args)
write_log("F1", args)
write_log(F1_all, args)
write_log("precision", args)
write_log(preci_all, args)
print("all subject test result:" + str(ACC))
print("all subject average result of ACC：" + str(np.mean(ACC)))
