from opt import *
from modeldefine import *
import time
from dataloader import *
from visualization import *
from termcolor import colored
import torch
import torch.backends.cudnn as cudnn
from checkpoint import *
import random
from trainer import *

def dataparallel(model, ngpus, gpu0=0):
    if ngpus == 0:
        assert False, "only support gpu mode"
    gpu_list = list(range(gpu0, gpu0+ngpus))
    # assert torch.cuda.device_count() >= gpu0+ngpus, "Invalid Number of GPUs"
    if isinstance(model, list):
        for i in range(len(model)):
            if ngpus >= 2:
                if not isinstance(model[i], nn.DataParallel):
                    model[i] = torch.nn.DataParallel(model[i], gpu_list).cuda()
            else:
                model[i] = model[i].cuda()
    else:
        if ngpus >= 2:
            if not isinstance(model, nn.DataParallel):
                model = torch.nn.DataParallel(model, gpu_list).cuda()
        else:
            model = model.cuda()
    return model


def getweights(layer, epoch_id, block_id, layer_id, log_writer):
    if isinstance(layer, nn.Conv2d):
        weights = layer.weight.data.cpu().numpy()
        weights_view = weights.reshape(weights.size)
        log_writer(input_data=weights_view, block_id=block_id,
                   layer_id=layer_id, epoch_id=epoch_id)


def main(net_opt=None):
    """requirements:
    apt-get install graphviz
    pip install pydot termcolor"""

    start_time = time.time()
    opt = net_opt or NetOption()

    # set torch seed
    # init random seed
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed(opt.manualSeed)
    cudnn.benchmark = True
    if opt.nGPU == 1 and torch.cuda.device_count() >= 1:
        assert opt.GPU <= torch.cuda.device_count()-1, "Invalid GPU ID"
        torch.cuda.set_device(opt.GPU)
    else:
        torch.cuda.set_device(opt.GPU)

    # create data loader
    data_loader = DataLoader(dataset=opt.data_set, data_path=opt.data_path, label_path=opt.label_path, batch_size=opt.batchSize,
                             n_threads=opt.nThreads, ten_crop=opt.tenCrop, dataset_ratio=opt.datasetRatio)
    train_loader, test_loader = data_loader.getloader()

    # define check point
    check_point = CheckPoint(opt=opt)
    # create residual network mode
    if opt.retrain:
        check_point_params = check_point.retrainmodel()
    elif opt.resume:
        check_point_params = check_point.resumemodel()
    else:
        check_point_params = check_point.check_point_params

    greedynet = None
    optimizer = check_point_params['opts']
    start_stage = check_point_params['stage'] or 0
    start_epoch = check_point_params['resume_epoch'] or 0
    if check_point_params['resume_epoch'] is not None:
        start_epoch += 1
    if start_epoch >= opt.nEpochs:
        start_epoch = 0
        start_stage += 1

    # model
    if opt.netType == 'ResNet':
        model = ResNetImageNet(
            opt=opt, num_classes=opt.nClasses, retrain=check_point_params['model'])
    elif opt.netType == 'DenseNet':
        model = DenseNetImageNet(
            opt=opt, num_classes=opt.nClasses, retrain=check_point_params['model'])
    elif opt.netType == 'Inception-v3':
        model = Inception3ImageNet(
            opt=opt, num_classes=opt.nClasses, retrain=check_point_params['model'])
    elif opt.netType == 'AlexNet':
        model = AlexNetImageNet(
            opt=opt, num_classes=opt.nClasses, retrain=check_point_params['model'])
    elif opt.netType == 'VGG':
        model = VGGImageNet(
            opt=opt, num_classes=opt.nClasses, retrain=check_point_params['model'])
    else:
        assert False, "invalid net type"

    
    model = dataparallel(model, opt.nGPU, opt.GPU)
    trainer = Trainer(model=model, opt=opt, optimizer=optimizer)
    print("|===>Create trainer")

    if opt.testOnly:
        trainer.test(epoch=0, test_loader=test_loader)
        return

    # define visualizer
    visualize = Visualization(opt=opt)
    visualize.writeopt(opt=opt)

    best_auc = 0
    start_epoch = opt.resumeEpoch
    for epoch in range(start_epoch, opt.nEpochs):
        train_auc, train_loss = trainer.train(
            epoch=epoch, train_loader=train_loader)
        test_auc, test_loss, test_acc, test_precision, test_recall, test_f1, test_gmean, tn, fp, fn, tp = trainer.test(
            epoch=epoch, test_loader=test_loader)

        # write and print result
        log_str = "%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%d\t%d\t%d\t%d\t" % (epoch, train_auc, train_loss, test_auc,
                                                                test_loss, test_acc, test_precision, test_recall, test_f1, test_gmean, tn, fp, fn,tp )
        visualize.writelog(log_str)
        best_flag = False
        if best_auc <= test_auc:
            best_auc = test_auc
            best_flag = True
            print(colored("# %d ==>Best Result is: AUC: %f\n" % (
                epoch, best_auc), "red"))
        else:
            print(colored("# %d ==>Best Result is: AUC: %f\n" % (
                epoch, best_auc), "blue"))
        
        # save check_point
        check_point.savemodel(epoch=epoch, model=trainer.model,
                              opts=trainer.optimzer, best_flag=best_flag)

    end_time = time.time()
    time_interval = end_time-start_time
    t_hours = time_interval//3600
    t_mins = time_interval % 3600 // 60
    t_sec = time_interval % 60
    t_string = "Running Time is: " + \
        str(t_hours) + " hours, " + str(t_mins) + \
        " minutes," + str(t_sec) + " seconds\n"
    print(t_string)



if __name__ == '__main__':
    # main()
    main_opt = NetOption()   ##class NetOption()
    main_opt.paramscheck()
    main(main_opt)
