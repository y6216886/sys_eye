import torch
import torch.nn as nn
from torch.autograd import Variable
from modeldefine import *
import torch.autograd
import time
import math
from graphgen import *
import numpy as np
from sklearn import metrics
import pandas as pd

single_train_time = 0
single_test_time = 0
single_train_iters = 0
single_test_iters = 0


def svb(m):
    eps = 0.5
    for layer in m.modules():
        if isinstance(layer, nn.Conv2d):
            w_size = layer.weight.data.size()
            layer_weight = (layer.weight.data.view(w_size[0], w_size[1]*w_size[2]*w_size[3])).cpu()
            U, S, V = torch.svd(layer_weight)
            S = S.clamp(1.0/(1+eps), 1+eps)
            layer_weight = torch.mm(torch.mm(U, torch.diag(S)), V.t())
            layer.weight.data.copy_(layer_weight.view(w_size[0], w_size[1], w_size[2]*w_size[3]))


def bbn(m):
    eps = 1.0
    for layer in m.modules():
        if isinstance(layer, nn.BatchNorm2d):
            std = torch.sqrt(layer.running_var+layer.eps)
            alpha = torch.mean(layer.weight.data/std)
            low_bound = (std*alpha/(1+eps)).cpu()
            up_bound = (std*alpha*(1+eps)).cpu()
            layer_weight_cpu = layer.weight.data.cpu()
            layer_weight = layer_weight_cpu.numpy()
            layer_weight.clip(low_bound.numpy(), up_bound.numpy())
            layer.weight.data.copy_(torch.Tensor(layer_weight))


def getlearningrate(epoch, opt):
    # update lr
    lr = opt.LR
    if opt.lrPolicy == "multistep":
        if epoch + 1.0 >= opt.nEpochs * opt.ratio[1]:  # 0.6 or 0.8
            lr = opt.LR * 0.01
        elif epoch + 1.0 >= opt.nEpochs * opt.ratio[0]:  # 0.4 or 0.6
            lr = opt.LR * 0.1
    elif opt.lrPolicy == "linear":
        k = (0.001-opt.LR)/math.ceil(opt.nEpochs/2.0)
        lr = k*math.ceil((epoch+1)/opt.step)+opt.LR
    elif opt.lrPolicy == "exp":
        power = math.floor((epoch+1)/opt.step)
        lr = lr*math.pow(opt.gamma, power)
    else:
        assert False, "invalid lr policy"

    return lr


def computetencrop(outputs, labels):
    output_size = outputs.size()
    outputs = outputs.view(output_size[0]/10, 10, output_size[1])
    outputs = outputs.sum(1).squeeze(1)
    # compute top1
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t()
    top1_count = pred.eq(labels.data.view(1, -1).expand_as(pred)).view(-1).float().sum(0)
    top1_error = 100.0 - 100.0 * top1_count / labels.size(0)
    top1_error = float(top1_error.cpu().numpy())

    # compute top5
    _, pred = outputs.topk(5, 1, True, True)
    pred = pred.t()
    top5_count = pred.eq(labels.data.view(1, -1).expand_as(pred)).view(-1).float().sum(0)
    top5_error = 100.0 - 100.0 * top5_count / labels.size(0)
    top5_error = float(top5_error.cpu().numpy())
    return top1_error, 0, top5_error


def computeresult(outputs, labels, loss, top5_flag=False):
    if isinstance(outputs, list):
        top1_loss = []
        top1_error = []
        top5_error = []
        for i in range(len(outputs)):
            # get index of the max log-probability
            predicted = outputs[i].data.max(1)[1]
            top1_count = predicted.ne(labels.data).cpu().sum()
            top1_error.append(100.0*top1_count/labels.size(0))
            top1_loss.append(loss[i].data[0])
            if top5_flag:
                _, pred = outputs[i].data.topk(5, 1, True, True)
                pred = pred.t()
                top5_count = pred.eq(labels.data.view(1, -1).expand_as(pred)).view(-1).float().sum(0)
                single_top5 = 100.0 - 100.0 * top5_count / labels.size(0)
                single_top5 = float(single_top5.cpu().numpy())
                top5_error.append(single_top5)

    else:
        # get index of the max log-probability
        predicted = outputs.data.max(1)[1]
        top1_count = predicted.ne(labels.data).cpu().sum()
        top1_error = 100.0*top1_count/labels.size(0)
        top1_loss = loss.data[0]
        top5_error = 100.0
        if top5_flag:
            _, pred = outputs.data.topk(5, 1, True, True)
            pred = pred.t()
            top5_count = pred.eq(labels.data.view(1, -1).expand_as(pred)).view(-1).float().sum(0)
            top5_error = 100.0 - 100.0 * top5_count/labels.size(0)
            top5_error = float(top5_error.cpu().numpy())

    if top5_flag:
        return top1_error, top1_loss, top5_error
    else:
        return top1_error, top1_loss, top5_error


def computeAUC(outputs, labels):
    if isinstance(outputs, list):
        pred = np.concatenate(outputs, axis=0)
        y = np.concatenate(labels, axis=0)
    else:
        pred = outputs
        y = labels
    fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    if np.isnan(auc):
        auc = 0
    return auc, fpr, tpr

def computeEval(outputs, labels):
    if isinstance(outputs, list):
        pred = np.concatenate(outputs, axis=0)
        y = np.concatenate(labels, axis=0)
    else:
        pred = outputs
        y = labels
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0
    # acc 
    acc = metrics.accuracy_score(y, pred)
    # tn, fp, fn, tp
    tn, fp, fn, tp = metrics.confusion_matrix(y, pred).ravel()
    # precision 
    precision = np.nan if (tp+fp) == 0 else float(tp)/(tp+fp)
    # recall
    recall = np.nan if (tp+fn) == 0 else float(tp)/(tp+fn)
    # F1
    f1 = metrics.f1_score(y, pred, pos_label=1, average='binary')
    # g-mean
    specificity = np.nan if (tn+fp) == 0 else float(tn)/(tn+fp)
    gmean =  math.sqrt(recall*specificity)
    print ("tn, fp, fn, tp")
    print (tn, fp, fn, tp)
    return acc, precision, recall, f1, gmean, tn, fp, fn, tp

def printresult(epoch, nEpochs, count, iters, lr, data_time, iter_time, loss, mode="Train"):
    global single_train_time, single_test_time
    global single_train_iters, single_test_iters

    log_str = ">>> %s [%.3d|%.3d], Iter[%.3d|%.3d], DataTime: %.4f, IterTime: %.4f, lr: %.4f" \
              % (mode, epoch + 1, nEpochs, count, iters, data_time, iter_time, lr)

    # compute cost time
    if mode == "Train":
        single_train_time = single_train_time*0.95 + 0.05*(data_time+iter_time)
        # single_train_time = data_time + iter_time
        single_train_iters = iters
    else:
        single_test_time = single_test_time*0.95 + 0.05*(data_time+iter_time)
        # single_test_time = data_time+iter_time
        single_test_iters = iters
    total_time = (single_train_time*single_train_iters+single_test_time*single_test_iters)*nEpochs
    time_str = ",Cost Time: %d Days %d Hours %d Mins %.4f Secs" % (total_time // (3600*24),
                                                                   total_time // 3600.0 % 24,
                                                                   total_time % 3600.0 // 60,
                                                                   total_time % 60)
    print (log_str+time_str)

def writeData(x, y):
    path_name = 'test.csv' 
    # df_new = pd.DataFrame({'fpr':x, 'tpr':y})
    # df = pd.read_csv(path_name)
    # result = pd.concat([df, df_new], axis=1)
    # result.to_csv(path_name, index=False)

def writeDiseaseType(x, y):
    path_name = 'disease_type.csv'
    df = pd.DataFrame({'realLabel':x, 'predictLabel': y})
    df['isTrue'] = df.apply(lambda x: ('True' if (x['realLabel'] == x['predictLabel']) else 'False'), axis=1)
    df.to_csv(path_name, index=False)


class Trainer(object):
    realLabelsarr = []
    predictLabelsarr = []
    def __init__(self, model, opt, optimizer=None):
        self.opt = opt
        self.model = model
        # print (model)
        if self.opt.trainingType == 'onevsall':
            self.criterion = nn.BCELoss().cuda()
        else:
            self.criterion = nn.CrossEntropyLoss().cuda()
        self.lr = self.opt.LR
        # self.optimzer = optimizer or torch.optim.RMSprop(self.model.parameters(),
        #                                              lr=self.lr,
        #                                              eps=1,
        #                                              momentum=self.opt.momentum,
        #                                              weight_decay=self.opt.weightDecay)
        self.optimzer = optimizer or torch.optim.SGD(self.model.parameters(),
                                                     lr=self.lr,
                                                     momentum=self.opt.momentum,
                                                     weight_decay=self.opt.weightDecay,
                                                     nesterov=True)


    def updateopts(self):
        self.optimzer = torch.optim.SGD(self.model.parameters(),
                                        lr=self.lr,
                                        momentum=self.opt.momentum,
                                        weight_decay=self.opt.weightDecay,
                                        nesterov=True)

    def updatelearningrate(self, epoch):
        self.lr = getlearningrate(epoch=epoch, opt=self.opt)
        # update learning rate of model optimizer
        for param_group in self.optimzer.param_groups:
            param_group['lr'] = self.lr

    def forward(self, images, labels=None):
        # forward and backward and optimize
        output = self.model(images)
        # print("output!!!",output.size())
        # print(output)
        # print("label!!!",labels.size())

        if labels is not None:
            loss = self.criterion(output, labels)
        else:
            loss = None
        return output, loss

    def backward(self, loss):
        self.optimzer.zero_grad()
        loss.backward()

        self.optimzer.step()

    def train(self, epoch, train_loader):
        loss_sum = 0
        iters = len(train_loader)
        output_list = []
        label_list = []

        self.updatelearningrate(epoch)
        
        self.model.train()

        start_time = time.time()
        end_time = start_time

        for i, (images, labels) in enumerate(train_loader):
            # if images.size(0) != self.opt.batchSize:
            #     break

            self.model.train()
            start_time = time.time()
            data_time = start_time - end_time

            # print("size0",labels.size(0))
            # print("size0", labels.size(0))
            target_disease = torch.LongTensor(labels.size(0)).zero_() + int(self.opt.disease_type)  ##disease_type 就是onevsall中的one代表的数据类型
            # print("targetdi",target_disease)
            # print("zero",torch.LongTensor(labels.size(0)).zero_())
            # print("diseatype", self.opt.disease_type)
            # print("labels", labels)
            reduce_labels = labels == target_disease
            reduce_labels = reduce_labels.type_as(images)   ##和images转为同一个数据格式
            # print("reduce_labels", reduce_labels)
            labels = reduce_labels
                
            images = images.cuda()

            labels = labels.cuda()##加入了gpu编号tensor([1., 1., 1., 0., 1., 1., 0., 1., 0., 1., 1., 1., 1., 1., 1., 0.],device='cuda:4')

            images_var = Variable(images)
            labels_var = Variable(labels)

            output, loss = self.forward(images_var, labels_var)
            # print("loss",loss)
            # print("loss.size[0]",loss.data[0])
            # print("loss.size", loss.data)
            prediction = output.data.cpu()
            # print("pred",prediction)

            output_list.append(prediction.numpy())
            label_list.append(reduce_labels.numpy())

            self.backward(loss)

            # loss_sum += loss.data[0]
            loss_sum += float(loss.data)
            #Here, total_loss is accumulating history across your training loop, since loss is a differentiable variable with autograd history.
            # You can fix this by writing total_loss += float(loss) instead.
            end_time = time.time()

            iter_time = end_time - start_time

            # printresult(epoch, self.opt.nEpochs, i+1, iters, self.lr, data_time, iter_time,
            #             loss.data[0], mode="Train")
            printresult(epoch, self.opt.nEpochs, i + 1, iters, self.lr, data_time, iter_time,
                        loss.data, mode="Train")
        loss_sum /= iters
        auc, fpr, tpr = computeAUC(output_list, label_list)
        # acc, precision, recall, f1, gmean = computeEval(output_list, label_list)
        print ("|===>Training AUC: %.4f Loss: %.4f "  % (auc, loss_sum))
        return auc, loss_sum



    def test(self, epoch, test_loader):
        loss_sum = 0
        iters = len(test_loader)
        output_list = []
        label_list = []

        self.model.eval()

        start_time = time.time()
        end_time = start_time
        for i, (images, labels) in enumerate(test_loader):
            # if images.size(0) != self.opt.batchSize:
            #     break

            start_time = time.time()
            data_time = start_time - end_time

            target_disease = torch.LongTensor(labels.size(0)).zero_() + int(self.opt.disease_type)
            reduce_labels = labels == target_disease
            reduce_labels = reduce_labels.type_as(images)
            labels = reduce_labels

            labels = labels.cuda()
            # labels_var = Variable(labels, volatile=True)
            with torch.no_grad():
                labels_var = Variable(labels)
            images = images.cuda()
            with torch.no_grad():
                images_var = Variable(images)
            # images_var = Variable(images, volatile=True)



            output, loss = self.forward(images_var, labels_var)
            # print("output",output)#
            # print("loss",loss)#

            prediction = output.data.cpu()
            output_list.append(prediction.numpy())
            label_list.append(reduce_labels.numpy())

            #loss_sum += loss.data[0]
            loss_sum += loss.data
            end_time = time.time()
            iter_time = end_time - start_time

            printresult(epoch, self.opt.nEpochs, i+1, iters, self.lr, data_time, iter_time,
                       # loss.data[0], mode="Test")
                        loss.data, mode="Test")

        loss_sum /= iters
        auc, fpr, tpr = computeAUC(output_list, label_list)
        acc, precision, recall, f1, gmean, tn, fp, fn, tp = computeEval(output_list, label_list)
        print ("|===>Testing AUC: %.4f Loss: %.4f acc: %.4f precision: %.4f recall: %.4f f1: %.4f gmean: %.4f"  % (auc, loss_sum, acc, precision, recall, f1, gmean))
        return auc, loss_sum, acc, precision, recall, f1, gmean, tn, fp, fn, tp




