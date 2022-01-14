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
from until.ConvNet import ShallowFBCSPNet
import until.lr_schedule as lr_schedule
from until.constraints import MaxNormDefaultConstraint
from until.preProcessTeachermodel import preprocess_teacher
import losses_supcon
import argparse
from saveResult import write_log
from until.loadData import *

parser = argparse.ArgumentParser(description='input the dataset dir path.')

parser.add_argument('--backbone_net_all', type=str, default=['shallow'], help='choose model: shallow')
parser.add_argument('--test_interval', type=int, default=1, help='iter')
parser.add_argument('--max_epochs', type=int, default=1500)
parser.add_argument('--folds', type=int, default=10, help='x fold cross validation')
parser.add_argument('--subject', type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9])  #
parser.add_argument('--log_path', type=str, default='./result/', help='folder')
parser.add_argument('--foldname', type=str, default='result.txt', help='filename')
parser.add_argument('--alpha1', type=float, default=1, help='parameter of co-contrastive loss')
parser.add_argument('--alpha2', type=float, default=0.001, help='parameter of mse_loss')
parser.add_argument('--batch_size', type=int, default=120)
parser.add_argument('--dataset', type=str, default="2a", help='choose: 2a, 2b')

args = parser.parse_args()

optim_dict = {"Adam": optim.Adam, "SGD": optim.SGD}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cuda = True

torch.cuda.set_device(0)

write_log(args.max_epochs, args)
write_log(args.subject, args)
write_log(args.dataset, args)
acc = list()
ACC = list()
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
            print("ten folds cross-validation: Subject=" + str(subject_id) + ", fold=" + str(fold))
            # train data
            traindata1 = trainingdataall_2[train_index]
            traindata_ema = trainingdataall_3[train_index]
            trainlabel1 = traininglabelall_2[train_index]
            trainlabel_ema = traininglabelall_3[train_index]

            traindata = torch.cat((traindata1, traindata_ema), dim=1)
            trainlabel = torch.cat((trainlabel1, trainlabel_ema), dim=0)

            X_train = Variable(torch.unsqueeze(traindata, dim=3))
            trainlabel = trainlabel.reshape((len(train_index) * 2, 1))
            l1, l2 = torch.split(trainlabel, len(train_index), dim=0)
            X_train_label = torch.cat([l1, l2], dim=1)
            trainset = TensorDataset(X_train, X_train_label)
            train_iterator = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)

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
                model_ema = ShallowFBCSPNet(n_chans, n_classes, input_time_length=input_time_length,
                                            final_conv_length='auto')
            preprocess_teacher(model, model_ema)
            if cuda:
                model.cuda()
                model_ema.cuda()

            model_constraint = MaxNormDefaultConstraint()

            cl_loss = nn.CrossEntropyLoss()
            MSE = nn.MSELoss()
            loss_shallow = losses_supcon.SupConLoss()
            loss_shallow_ema = losses_supcon.SupConLoss()

            # metrics
            test_accuracy = np.zeros(shape=[0], dtype=float)

            parameter_list = [{"params": model.parameters()}]
            optimizer = optim_dict["Adam"](parameter_list, **({"lr": 0.001, "weight_decay": 0}))

            schedule_param = {"init_lr": 0.001, "gamma": 10, "power": 0.75}
            lr_scheduler = lr_schedule.schedule_dict["inv"]

            for i in trange(args.max_epochs):
                model.train()
                optimizer = lr_scheduler(optimizer, i / args.max_epochs, **schedule_param)

                for step, batch in enumerate(train_iterator):
                    inputs, labels = batch
                    optimizer.zero_grad()

                    input_vars, labels_vars = Variable(inputs).cuda().float(), Variable(labels).cuda().long()

                    input_vars, input_vars_ema = torch.split(input_vars, n_chans, dim=1)
                    labels_vars, labels_vars_ema = torch.split(labels_vars, 1, dim=1)

                    input = torch.cat((input_vars, input_vars_ema), dim=0)

                    preds, feature, _ = model(input)
                    preds_ema, feature_ema, _ = model_ema(input)

                    preds_ema = preds_ema.detach()
                    feature_ema = feature_ema.detach()

                    l2_norm = torch.sqrt(torch.sum(feature ** 2, dim=1))
                    features = feature / torch.unsqueeze(l2_norm, dim=1)
                    l2_norm_2 = torch.sqrt(torch.sum(feature_ema ** 2, dim=1))
                    feature_ema = feature_ema / torch.unsqueeze(l2_norm_2, dim=1)

                    feature_ori, feature_aug = torch.split(features, int(input.shape[0] / 2), dim=0)
                    feature_ema_ori, feature_ema_aug = torch.split(feature_ema, int(input.shape[0] / 2), dim=0)

                    feature_cat1 = torch.cat(
                        [torch.unsqueeze(feature_ori, dim=1), torch.unsqueeze(feature_ema_aug, dim=1)], dim=1)
                    feature_cat2 = torch.cat(
                        [torch.unsqueeze(feature_aug, dim=1), torch.unsqueeze(feature_ema_ori, dim=1)], dim=1)

                    loss1 = loss_shallow(feature_cat1, labels_vars, epoch=i, step=step)
                    loss2 = loss_shallow_ema(feature_cat2, labels_vars_ema, epoch=i, step=step)

                    preds_ema_ori, preds_ema_aug = torch.split(preds_ema, int(input.shape[0] / 2), dim=0)
                    preds_teac = torch.cat([preds_ema_aug, preds_ema_ori], dim=0)
                    loss_mse = MSE(preds, preds_teac)

                    labels_vars = torch.cat((labels_vars, labels_vars), dim=0)
                    labels_vars_size = int(labels_vars.shape[0])
                    loss3 = cl_loss(preds, labels_vars.reshape((labels_vars_size)))

                    alpha1 = (2. / (1. + np.exp(-10 * i / args.max_epochs)) - 1) * args.alpha1
                    alpha2 = (2. / (1. + np.exp(-10 * i / args.max_epochs)) - 1) * args.alpha2

                    loss = loss3 + alpha2 * loss_mse + alpha1 * (loss1 + loss2)

                    loss.backward()
                    optimizer.step()

                    moment_update_ema(model, model_ema, 0.999)

                    if model_constraint is not None:
                        model_constraint.apply(model)

            with torch.no_grad():
                model.eval()
                # test in the train
                b_x = X_test.float()
                b_y = testinglabel.long()
                input_vars = np_to_var(b_x, pin_memory=False).float()
                labels_vars_test = np_to_var(b_y, pin_memory=False).type(torch.LongTensor)
                input_vars = input_vars.cuda()
                labels_vars_test = labels_vars_test.cuda()

                outputs, f, _ = model(input_vars)

                y_test_pred = torch.max(outputs, 1)[1].cpu().data.numpy().squeeze()

                acc = metrics.accuracy_score(labels_vars_test.cpu().data.numpy(), y_test_pred)

                test_accuracy = np.append(test_accuracy, acc)

        # save ten-folds result
        acc_result = test_accuracy[-10:]

        print("S=" + str(subject_id))
        print("all 10 folds ACC:", acc_result)
        s = "kf: 1-10"
        write_log("S = " + str(subject_id), args)
        write_log(s, args)
        write_log("ACC", args)
        write_log(list(acc_result), args)

        ACC.append(acc_result.mean())

        acc = list()

write_log("all test subject result：", args)
write_log("ACC", args)
print(ACC)
print("all subject average result of ACC：" + str(np.mean(ACC)))
