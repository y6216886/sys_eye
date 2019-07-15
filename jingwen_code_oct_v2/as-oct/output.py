# import torch
# from modeldefine import *
# from opt import *
# from checkpoint import *
# from dataloader import *
# ##model.load_state_dict(torch.load("params.pkl"))
# opt = NetOption()
# check_point = CheckPoint(opt=opt)
# model = ResNetImageNet(
#             opt=opt, num_classes=opt.nClasses, retrain=check_point.check_point_params['model'])
# model.load_state_dict(torch.load('/home/yangyifan/code1/model/net_032.pth'))
#
# data_loader = DataLoader(dataset=opt.data_set, data_path=opt.data_path, label_path=opt.label_path, batch_size=opt.batchSize,
#                              n_threads=opt.nThreads, ten_crop=opt.tenCrop, dataset_ratio=opt.datasetRatio)
# train_loader, test_loader = data_loader.getloader()
#
# for i, (images, labels) in enumerate(test_loader):
#     images = images.cuda()
#     images_var = Variable(images)
#     output = model(images_var)
#     print("output",output.size(),"labels",labels)


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
    gpu_list = list(range(gpu0, gpu0 + ngpus))
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
        assert opt.GPU <= torch.cuda.device_count() - 1, "Invalid GPU ID"
        torch.cuda.set_device(opt.GPU)
    else:
        torch.cuda.set_device(opt.GPU)

    # create data loader
    data_loader = DataLoader(dataset=opt.data_set, data_path=opt.data_path, label_path=opt.label_path,
                             batch_size=opt.batchSize,
                             n_threads=opt.nThreads, ten_crop=opt.tenCrop, dataset_ratio=opt.datasetRatio)
    val_loader= data_loader.getloader()

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
    ####v3
    modelList = []
    print("111",opt.retrain)
    if opt.data_set == "imagenet" or opt.data_set == "quality" or opt.data_set == "multi" or opt.data_set == "validation" or opt.data_set == 'asoct':
        for i in range(len(opt.retrain)):
            ####v3
            if opt.netType == 'ResNet':
                model = ResNetImageNet(
                    opt=opt, num_classes=opt.nClasses, retrain=check_point_params['model'][i])
            elif opt.netType == 'DenseNet':
                model = DenseNetImageNet(
                    opt=opt, num_classes=opt.nClasses, retrain=check_point_params['model'][i])
            elif opt.netType == 'Inception-v3':
                model = Inception3ImageNet(
                    opt=opt, num_classes=opt.nClasses, retrain=check_point_params['model'][i])
            elif opt.netType == 'AlexNet':
                model = AlexNetImageNet(
                    opt=opt, num_classes=opt.nClasses, retrain=check_point_params['model'][i])
            elif opt.netType == 'VGG':
                model = VGGImageNet(
                    opt=opt, num_classes=opt.nClasses, retrain=check_point_params['model'][i])
            else:
                assert False, "invalid net type"
            modelList.append(model)  ##网络结构的列表v3
    else:  ####v3
        assert False, "invalid data set"  ###v3

    modelList = dataparallel(modelList, opt.nGPU, opt.GPU)

    model = dataparallel(model, opt.nGPU, opt.GPU)
    res_dict = {'fileName': [], 'Type': [], 'realLabel': [], 'predLabel': []}
    for image, realLabel, imgId, imgType in val_loader:
        print(imgId)
        image=image.cuda()
        image_var = Variable(image)
        res_list = []
        for (i, m) in enumerate(modelList):
            m.eval()
            p = m(image_var).data[0][0]
            res_list.append(p)
        pred = res_list.index(max(res_list)) + 1
        res_dict['fileName'].append(imgId[0])
        res_dict['Type'].append(imgType[0])
        res_dict['realLabel'].append(realLabel.cpu().numpy()[0])
        res_dict['predLabel'].append(pred)

    # print res_dict
    res_df = pd.DataFrame(res_dict)
    # print res_df
    save_path = '/home/yangyifan/jingwen_code_oct_v2/as-oct/pred_train.csv'
    res_df.to_csv(save_path)


if __name__ == '__main__':
    # main()
    main_opt = NetOption()  ##class NetOption()
    main_opt.paramscheck()
    main(main_opt)
