import argparse
import csv
import json
import os
import os.path as osp
import statistics
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import helper_utils.network as network
import helper_utils.loss as loss
import helper_utils.pre_process as prep
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
import helper_utils.lr_schedule as lr_schedule
import helper_utils.data_list
from torch.autograd import Variable
import random
import pdb
import math
from helper_utils.data_list import ImageList

from helper_utils.logger import Logger

from helper_utils.EarlyStopping import EarlyStopping

from helper_utils.sampler import ImbalancedDatasetSampler
from datetime import datetime

seed = 1000

now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
dt_string = dt_string.replace("/", "_").replace(" ", "_").replace(":", "_").replace(".", "_")
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'

torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def image_classification_test(loader, model, test_10crop=False, num_iterations=0):
    start_test = True
    with torch.no_grad():
        if test_10crop:
            iter_test = [iter(loader['test'][i]) for i in range(1)]
            for i in range(len(loader['test'][0])):
                data = [iter_test[j].next() for j in range(1)]
                inputs = [data[j][0] for j in range(1)]
                labels = data[0][1]
                for j in range(1):
                    inputs[j] = inputs[j].cuda()
                labels = labels
                outputs = []
                for j in range(1):
                    _, predict_out = model(inputs[j])
                    outputs.append(nn.Softmax(dim=1)(predict_out))
                outputs = sum(outputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    # all_label = torch.cat((all_label, labels.char()), 0)
                    all_label = all_label + labels
                    # print(labels)


        else:
            iter_test = iter(loader["test"])
            for i in range(len(loader['test'])):
                data = iter_test.next()
                inputs = data[0]
                labels = data[1]
                inputs = inputs.cuda()
                labels = labels.cuda()
                _, outputs = model(inputs)
                if start_test:
                    all_output = outputs.float().cpu()
                    all_label = labels.float()
                    start_test = False
                else:
                    all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                    all_label = torch.cat((all_label, labels.float()), 0)
    # print(len(all_output))
    # print(len(all_label))

    all_output_numpy = all_output.numpy()

    with open(config["logs_path"] + '/_confidence_values_.csv', mode='w') as file:
        csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for value in range(len(all_output_numpy)):
            csv_writer.writerow([all_label[value], all_output_numpy[value][0], all_output_numpy[value][1]])

    _, predict = torch.max(all_output, 1)

    all_values_CSV = []

    predict_numpy = predict.numpy()

    for ins in range(len(all_label)):
        main_image_label = all_label[ins].split(',')
        # label[1]
        all_values_CSV.append([main_image_label[1], predict_numpy[ins]])

    # print(all_values_CSV)

    # print(torch.squeeze(predict))
    main_image_results = []

    log_good_count = 0
    log_good_per = 0

    import itertools
    for key, images in itertools.groupby(all_values_CSV, lambda x: x[0]):
        # print(key, images)

        count0 = 0
        count1 = 0

        total = 0
        all_images = []

        for each_img in images:
            all_images.append(each_img)
            total = total + 1
            # print(each_img[2])

            if (each_img[1] == 0):
                count0 = count0 + 1
            elif (each_img[1] == 1):
                count1 = count1 + 1

        zero_per = count0 / total * 100
        one_per = count1 / total * 100

        good_per_arr = []
        if len(all_images) > 200:
            for k in range(50):

                random_200_rsulkts = random.sample(all_images, 200)

                good = 0
                bad = 0

                all_sperms = 0
                for pred in random_200_rsulkts:

                    all_sperms = all_sperms + 1
                    # print(each_img[2])

                    if (pred[1] == 0):
                        good = good + 1
                    elif (pred[1] == 1):
                        bad = bad + 1
                good_per = good / all_sperms * 100
                bad_per = bad / all_sperms * 100

                # print(good_per, bad_per, good, bad, all_sperms)

                good_per_arr.append(good_per)
                # print("good",good_per)

            sd = statistics.stdev(good_per_arr)
            mean = statistics.mean(good_per_arr)


        else:
            sd = 0
            mean = zero_per


        # print("mean", mean, "sd", sd)


        true_value_slide = 'not found'
        with open('../../data/sd1/phone_doctor_final_new.csv') as csv_file:
            csv_reader = csv.reader(csv_file,delimiter = ',')

            for row in csv_reader:

                if row[0] == key:
                    # print(key, row[0])

                    true_value_slide = row[1]



        main_image_results.append([num_iterations, key, zero_per, one_per, count0, count1, total, true_value_slide, sd, mean])
        print(num_iterations, key, zero_per, one_per, count0, count1, total, true_value_slide)
        log_good_count = count0
        log_good_per = zero_per

    # with open(config["output_path"] + '/_'+str(num_iterations)+'_each_image_results_sperm.csv', 'w') as f:
    with open(config["logs_path"] + '/_each_image_results_sperm.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerows(main_image_results)

    f.close()

    return log_good_count, log_good_per, total

    # exit()

    # print(predict)
    # print(all_label)
    #
    # print(torch.squeeze(predict))
    #
    # print(all_label.size()[0]
    #       )
    #

    # exit()
    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # cm = confusion_matrix(all_label, torch.squeeze(predict).float())

    # print(cm)

    # exit()
    # return no


def train(config):
    now = dt_string.replace(" ", "_").replace(":", "_").replace(".", "_")
    logger = Logger(config["logs_path"] + "tensorboard/" + now)
    model_path = osp.join(config["output_path"], "best_model.pth.tar")
    early_stopping = EarlyStopping(patience=200, verbose=True, model_path=model_path)

    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params'])
    prep_dict["target"] = prep.image_train(**config["prep"]['params'])
    if prep_config["test_10crop"]:
        prep_dict["test"] = prep.image_test_10crop(**config["prep"]['params'])
    else:
        prep_dict["test"] = prep.image_test(**config["prep"]['params'])

    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]
    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(), \
                                transform=prep_dict["source"], data="source")

    dset_loaders["source"] = DataLoader(dsets["source"], sampler=ImbalancedDatasetSampler(dsets["source"]),
                                        batch_size=train_bs, \
                                        shuffle=False, num_workers=4, drop_last=True)

    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(), \
                                transform=prep_dict["target"], data="target")

    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, \
                                        shuffle=False, num_workers=4, drop_last=True)

    if prep_config["test_10crop"]:
        for i in range(1):
            dsets["test"] = [ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                       transform=prep_dict["test"][i], data="target") for i in range(1)]
            dset_loaders["test"] = [DataLoader(dset, batch_size=test_bs, \
                                               shuffle=False, num_workers=4) for dset in dsets['test']]
    else:
        dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(), \
                                  transform=prep_dict["test"], data="target")
        dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                          shuffle=False, num_workers=4)

    class_num = config["network"]["params"]["class_num"]
    print(class_num)

    ## set base network
    net_config = config["network"]
    base_network = net_config["name"](**net_config["params"])
    base_network = base_network.cuda()

    ## add additional network for some methods
    if config["loss"]["random"]:
        random_layer = network.RandomLayer([base_network.output_num(), class_num], config["loss"]["random_dim"])
        ad_net = network.AdversarialNetwork(config["loss"]["random_dim"], 1024)
    else:
        random_layer = None
        ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    if config["loss"]["random"]:
        random_layer.cuda()
    ad_net = ad_net.cuda()
    parameter_list = base_network.get_parameters() + ad_net.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, \
                                         **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        ad_net = nn.DataParallel(ad_net, device_ids=[int(i) for i in gpus])
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    ## train
    len_train_source = len(dset_loaders["source"])
    len_train_target = len(dset_loaders["target"])
    transfer_loss_value = classifier_loss_value = total_loss_value = 0.0
    best_acc = 0.0
    best_loss = best_total_loss_numpy = best_transfer_loss_numpy = best_classifier_loss_numpy = 10000
    train_time_start = time.time()
    log_good_count = log_total_sperms = log_good_per = 0.0
    start_time = time.time()
    for i in range(config["num_iterations"]):

        if i % config["test_interval"] == config["test_interval"] - 1:
            itr_log = "num_iterations  " + str(i)

            config["out_file"].write(itr_log + "\n")

            config["out_file"].flush()
            print(itr_log)

            train_time_end = time.time()

            test_time_start = time.time()
            base_network.train(False)
            log_good_count, log_good_per, log_total_sperms = image_classification_test(dset_loaders, \
                                                                                       base_network,
                                                                                       test_10crop=prep_config[
                                                                                           "test_10crop"],
                                                                                       num_iterations=i)
            temp_model = nn.Sequential(base_network)
            # if temp_acc > best_acc:
            #     best_acc = temp_acc
            best_model = temp_model

            test_time_end = time.time()

            log_str = "iter: {:05d}, precision:".format(i)
            testing_time = test_time_end - test_time_start
            log_test_time = "testing time:" + str(testing_time)

            training_time = train_time_start - train_time_end
            log_train_time = "training time for:" + str(i) + "iterations: " + str(training_time)
            config["out_file"].write(log_str + "\n")
            config["out_file"].write(log_test_time + "\n")
            config["out_file"].write(log_train_time + "\n")

            config["out_file"].write("\n")

            config["out_file"].flush()
            print(log_str)
            print(log_test_time)
            print(log_train_time)

            train_time_start = time.time()
            print()
            print()

        # if i % config["snapshot_interval"] == 0:
        #     print("snapshot")
        # torch.save(nn.Sequential(base_network), osp.join(config["logs_path"], \
        #                                                  "iter_{:05d}_model.pth.tar".format(i)))
        loss_params = config["loss"]
        ## train one iter
        base_network.train(True)
        ad_net.train(True)
        optimizer = lr_scheduler(optimizer, i, **schedule_param)
        optimizer.zero_grad()
        if i % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])
        if i % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])
        inputs_source, labels_source = iter_source.next()
        inputs_target, labels_target = iter_target.next()
        inputs_source, inputs_target, labels_source = inputs_source.cuda(), inputs_target.cuda(), labels_source.cuda()
        features_source, outputs_source = base_network(inputs_source)
        features_target, outputs_target = base_network(inputs_target)
        features = torch.cat((features_source, features_target), dim=0)
        outputs = torch.cat((outputs_source, outputs_target), dim=0)
        softmax_out = nn.Softmax(dim=1)(outputs)
        if config['method'] == 'CDAN+E':
            entropy = loss.Entropy(softmax_out)
            transfer_loss, weight_aware_entropy = loss.CDAN([features, softmax_out], ad_net, entropy,
                                                            network.calc_coeff(i), random_layer)
        elif config['method'] == 'CDAN':
            transfer_loss = loss.CDAN([features, softmax_out], ad_net, None, None, random_layer)
        elif config['method'] == 'DANN':
            transfer_loss = loss.DANN(features, ad_net)
        else:
            raise ValueError('Method cannot be recognized.')
        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)
        total_loss = loss_params["trade_off"] * transfer_loss + classifier_loss

        transfer_loss_numpy = transfer_loss.clone().cpu().detach().numpy()
        classifier_loss_numpy = classifier_loss.clone().cpu().detach().numpy()
        total_loss_numpy = total_loss.clone().cpu().detach().numpy()
        weight_aware_entropy_numpy = weight_aware_entropy.clone().cpu().detach().numpy()

        info = {'total_loss': total_loss_numpy,
                'classifier_loss': classifier_loss_numpy, 'transfer_loss': transfer_loss_numpy,
                'accuracy': log_good_per, 'entropy': weight_aware_entropy_numpy}

        for tag, value in info.items():
            logger.scalar_summary(tag, value, i)

        with open(config["logs_path"] + '/loss_values_.csv', mode='a') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            csv_writer.writerow(
                [i, total_loss_numpy, transfer_loss_numpy, classifier_loss_numpy, weight_aware_entropy_numpy])
        total_loss.backward()
        optimizer.step()

        temp_model = nn.Sequential(base_network)
        temp_loss = total_loss_numpy
        if temp_loss < best_loss:
            best_loss = temp_loss
            best_model = temp_model
            best_itr = i

            best_classifier_loss_numpy = classifier_loss_numpy
            best_total_loss_numpy = total_loss_numpy
            best_transfer_loss_numpy = transfer_loss_numpy

        early_stopping(total_loss, nn.Sequential(base_network))

        # print(i)

        if early_stopping.early_stop:
            print("Early stopping")

            # torch.save(nn.Sequential(base_network), osp.join(config["model_path"], "best_model.pth.tar"))

            break

    base_network.train(False)

    torch.save(best_model, osp.join(config["model_path"], "best_model" + str(best_itr) + ".pth.tar"))

    base_network = torch.load(osp.join(config["model_path"], "best_model" + str(best_itr) + ".pth.tar"))

    log_good_count, log_good_per, log_total_sperms = image_classification_test(dset_loaders, \
                                                                               base_network,
                                                                               test_10crop=prep_config[
                                                                                   "test_10crop"],
                                                                               num_iterations=best_itr)

    print(log_good_per, log_good_count, log_total_sperms)

    config["out_file"].write(str(log_good_per) + "," + str(log_good_count) + "," + str(log_total_sperms) + "\n")
    config["out_file"].write("\n")

    config["out_file"].write(
        str(best_total_loss_numpy) + "," + str(best_transfer_loss_numpy) + "," + str(best_classifier_loss_numpy) + "\n")

    config["out_file"].flush()
    best_acc = log_good_per

    # config_array = []
    # return best_acc
    # for key, val in config.items():
    #     config_array.append(key + ":" + str(val))
    #
    #



    config_array = ["trail-" + str(trial_number),  best_classifier_loss_numpy ,best_transfer_loss_numpy,best_total_loss_numpy
                    , log_good_per,
                    log_good_count,
                    log_total_sperms , best_itr] + training_parameters
    config["trial_parameters_log"].writerow(config_array)

    return best_acc


if __name__ == "__main__":
    log_output_dir_root = '../../logs/sd4_on_sd1_patient/'
    results_output_dir_root = '../../experimental results/sd4_on_sd1_patient/'
    models_output_dir_root = '../../models/sd4_on_sd1_patient/'

    targets = [
               '',
               ]

    # target_path = '../../data/sd1/sd1_all_patient.txt'

    for _ in targets:

        trial_number = str(17) + "_" + dt_string

        source_path = '../../data/sd4/sd4_all_data_source.txt'
        # target_path = '../../data/sd1/sd1_all_patient_old.txt'
        target_path = '../../data/sd1/sd1_all_patient_common_remove_D940.txt'

        net = 'Xception'
        dset = 'sperm'

        lr_ = 0.01
        gamma = 0.001
        power = 0.75
        momentum = 0.9
        weight_decay = 0.0005
        nesterov = True
        optimizer = optim.SGD

        config = {}
        config['method'] = 'CDAN+E'
        config["gpu"] = '0,1,2'
        config["num_iterations"] = 7000
        config["test_interval"] = 5000
        batch_size =    64
        batch_size_test= 64
        use_bottleneck = False
        bottleneck_dim =512
        adv_lay_random = True
        random_dim = 1024
        no_of_classes = 2
        new_cls = True
        batchnom1 = True
        dropout1 = True

        bottleneck2 = True
        batchnom2 = True
        dropout2 = True


        header_list = ["trail no " ,  "best_classifier_loss" ,"best_transfer_loss","best_total_loss"
                    , "log_good_per",
                    "log_good_count",
                    "log_total_sperms" , "best_itr"] + \
                    [ "lr", "gamma", "power", "momentum", "weight_decay", "nesterov", "optimizer",
                               "batch_size", "batch_size_test", "use_bottleneck", "bottleneck_dim", "adv_lay_random", "random_dim",
                               "no_of_classes", "new_cls", "dset", "net", "source_path", "target_path", "output_path", "model_path"
                               ,"logs_path" , "gpu", "test_interval"]


        log_output_path = log_output_dir_root + net + '/' + 'trial-' + trial_number + '/seed_' + str(seed) + '/'
        trial_results_path = net + '/trial-' + trial_number + '/seed_' + str(seed) + '/'
        config["output_path"] = results_output_dir_root + trial_results_path
        config["model_path"] = models_output_dir_root + trial_results_path
        config["logs_path"] = log_output_path
        if not os.path.exists(config["logs_path"]):
            os.makedirs(config["logs_path"])
        if not os.path.exists(config["model_path"]):
            os.makedirs(config["model_path"])

        if not os.path.isfile(osp.join(log_output_dir_root, "log.csv")):
            with open(osp.join(log_output_dir_root, "log.csv"), mode='w') as param_log_file:
                param_log_writer = csv.writer(param_log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                param_log_writer.writerow(header_list)

        config["out_file"] = open(osp.join(config["logs_path"], "log.txt"), "w")


        config["trial_parameters_log"] = csv.writer(open(osp.join(log_output_dir_root, "log.csv"), "a"))



        config["prep"] = {"test_10crop": True, 'params': {"resize_size": 224, "crop_size": 224, 'alexnet': False}}
        config["loss"] = {"trade_off": 1.0}

        if "Xception" in net:
            config["network"] = \
                {"name": network.XceptionFc,
                 "params":
                     {"use_bottleneck": use_bottleneck,
                      "bottleneck_dim": bottleneck_dim,
                      "new_cls": new_cls}}

        config["loss"]["random"] = adv_lay_random
        config["loss"]["random_dim"] = random_dim

        config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': lr_, "momentum": momentum,
                                                                   "weight_decay": weight_decay, "nesterov": nesterov},
                               "lr_type": "inv",
                               "lr_param": {"lr": lr_, "gamma": gamma, "power": power}}


        # config["optimizer"] = {"type": optim.Adam, "optim_params": {'lr': lr_,
        #                                                            "weight_decay": weight_decay},
        #                        "lr_type": "inv",
        #                        "lr_param": {"lr": lr_, "gamma": gamma, "power": power}}

        config["dataset"] = dset
        config["data"] = {"source": {"list_path": source_path, "batch_size": batch_size},
                          "target": {"list_path": target_path, "batch_size": batch_size},
                          "test": {"list_path": target_path, "batch_size": batch_size_test}}
        config["optimizer"]["lr_param"]["lr"] = lr_
        config["network"]["params"]["class_num"] = no_of_classes

        config["out_file"].write(str(config))
        config["out_file"].flush()

        training_parameters = [ lr_, gamma, power, momentum, weight_decay, nesterov, optimizer,
                               batch_size, batch_size_test, use_bottleneck, bottleneck_dim, adv_lay_random, random_dim,
                               no_of_classes, new_cls, dset, net, source_path, target_path, config["output_path"], config["model_path"]
                               , config["logs_path"] , config["gpu"], config["test_interval"],"batchnorm1 added"]
        train(config)
