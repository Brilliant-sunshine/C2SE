import scipy.io
import torch
import numpy as np


# load original data(preprocessed)
def originalData(datafolder_train, datafolder_test, subject_id):
    all_data1 = scipy.io.loadmat(datafolder_train)
    trainingdata_1 = np.transpose(all_data1['data' + str(subject_id) + '_X'], (2, 1, 0))
    traininglabel_1 = np.squeeze(all_data1['data' + str(subject_id) + '_y'])

    all_data = scipy.io.loadmat(datafolder_test)
    testingdata_1 = np.transpose(all_data['data' + str(subject_id) + '_X'], (2, 1, 0))
    testinglabel_1 = np.squeeze(all_data['data' + str(subject_id) + '_y'])

    trainingdataall_1 = torch.cat((torch.from_numpy(trainingdata_1), torch.from_numpy(testingdata_1)),
                                  dim=0)
    traininglabelall_1 = torch.cat((torch.from_numpy(traininglabel_1), torch.from_numpy(testinglabel_1)),
                                   dim=0)
    return trainingdataall_1, traininglabelall_1

# load A'' data
def mixcatData(datafolder_train, datafolder_test, subject_id):
    all_data2 = scipy.io.loadmat(datafolder_train)
    trainingdata_2 = np.transpose(all_data2['data' + str(subject_id) + '_X'], (2, 1, 0))
    traininglabel_2 = np.squeeze(all_data2['data' + str(subject_id) + '_y'])

    all_data3 = scipy.io.loadmat(datafolder_test)
    testingdata_2 = np.transpose(all_data3['data' + str(subject_id) + '_X'], (2, 1, 0))
    testinglabel_2 = np.squeeze(all_data3['data' + str(subject_id) + '_y'])

    trainingdataall_2 = torch.cat((torch.from_numpy(trainingdata_2), torch.from_numpy(testingdata_2)),
                                  dim=0)
    traininglabelall_2 = torch.cat((torch.from_numpy(traininglabel_2), torch.from_numpy(testinglabel_2)),
                                   dim=0)
    return trainingdataall_2, traininglabelall_2

# load A data
def cutmixData(datafolder_train, datafolder_test, subject_id):
    all_data4 = scipy.io.loadmat(datafolder_train)
    trainingdata_3 = np.transpose(all_data4['data' + str(subject_id) + '_X'], (2, 1, 0))
    traininglabel_3 = np.squeeze(all_data4['data' + str(subject_id) + '_y'])

    all_data5 = scipy.io.loadmat(datafolder_test)
    testingdata_3 = np.transpose(all_data5['data' + str(subject_id) + '_X'], (2, 1, 0))
    testinglabel_3 = np.squeeze(all_data5['data' + str(subject_id) + '_y'])

    trainingdataall_3 = torch.cat((torch.from_numpy(trainingdata_3), torch.from_numpy(testingdata_3)),
                                  dim=0)
    traininglabelall_3 = torch.cat((torch.from_numpy(traininglabel_3), torch.from_numpy(testinglabel_3)),
                                   dim=0)
    return trainingdataall_3, traininglabelall_3

def processLabel(traininglabelall_1, traininglabelall_2, traininglabelall_3):
    traininglabelall_1 = torch.where(traininglabelall_1 == 1, torch.full_like(traininglabelall_1, 0),
                                     traininglabelall_1)
    traininglabelall_1 = torch.where(traininglabelall_1 == 2, torch.full_like(traininglabelall_1, 1),
                                     traininglabelall_1)
    traininglabelall_1 = torch.where(traininglabelall_1 == 3, torch.full_like(traininglabelall_1, 2),
                                     traininglabelall_1)
    traininglabelall_1 = torch.where(traininglabelall_1 == 4, torch.full_like(traininglabelall_1, 3),
                                     traininglabelall_1)

    traininglabelall_2 = torch.where(traininglabelall_2 == 1, torch.full_like(traininglabelall_2, 0),
                                     traininglabelall_2)
    traininglabelall_2 = torch.where(traininglabelall_2 == 2, torch.full_like(traininglabelall_2, 1),
                                     traininglabelall_2)
    traininglabelall_2 = torch.where(traininglabelall_2 == 3, torch.full_like(traininglabelall_2, 2),
                                     traininglabelall_2)
    traininglabelall_2 = torch.where(traininglabelall_2 == 4, torch.full_like(traininglabelall_2, 3),
                                     traininglabelall_2)

    traininglabelall_3 = torch.where(traininglabelall_3 == 1, torch.full_like(traininglabelall_3, 0),
                                     traininglabelall_3)
    traininglabelall_3 = torch.where(traininglabelall_3 == 2, torch.full_like(traininglabelall_3, 1),
                                     traininglabelall_3)
    traininglabelall_3 = torch.where(traininglabelall_3 == 3, torch.full_like(traininglabelall_3, 2),
                                     traininglabelall_3)
    traininglabelall_3 = torch.where(traininglabelall_3 == 4, torch.full_like(traininglabelall_3, 3),
                                     traininglabelall_3)
    return traininglabelall_1, traininglabelall_2, traininglabelall_3