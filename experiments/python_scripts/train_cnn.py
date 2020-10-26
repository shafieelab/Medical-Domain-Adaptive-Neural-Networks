import argparse
import copy
import csv
import os
import os.path as osp
import statistics
import time
from collections import OrderedDict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import helper_utils.network as network
# import helper_utils.pre_process as prep
import helper_utils.pre_process_old as prep
import yaml

from torch.utils.data import DataLoader
import helper_utils.lr_schedule as lr_schedule
from helper_utils.data_list_m import ImageList

import argparse

from helper_utils.logger import Logger
from helper_utils.sampler import ImbalancedDatasetSampler

from helper_utils.EarlyStopping import EarlyStopping
from helper_utils.tools import testing_sperm_slides, validation_loss, calc_transfer_loss, Entropy


def data_setup(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["source"] = prep.image_train(**config["prep"]['params_source'])
    prep_dict["test"] = prep.image_test(**config["prep"]['params_test'])
    prep_dict["valid_source"] = prep.image_test(**config["prep"]['params_source'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["source"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]

    dsets["source"] = ImageList(open(data_config["source"]["list_path"]).readlines(),
                                transform=prep_dict["source"])
    dset_loaders["source"] = DataLoader(dsets["source"], batch_size=train_bs,
                                        sampler=ImbalancedDatasetSampler(dsets["source"]),
                                        shuffle=False, num_workers=4, drop_last=True)

    dsets["valid_source"] = ImageList(open(data_config["valid_source"]["list_path"]).readlines(),
                                      transform=prep_dict["valid_source"])
    dset_loaders["valid_source"] = DataLoader(dsets["valid_source"], batch_size=test_bs,
                                              shuffle=False, num_workers=4)

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(),
                              transform=prep_dict["test"], labelled=data_config["test"]["labelled"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                      shuffle=False, num_workers=4)

    return dset_loaders


def network_setup(config):
    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]
    if config['dataset'] == 'malaria' and config['trained_model_path']:
        base_network = torch.load(config['trained_model_path'])[0]

        layers = [name.replace('.weight', '').replace('.bias', '') for name, _ in base_network.named_parameters()]
        layers_names = OrderedDict.fromkeys(layers)
        layers_freeze = list(layers_names)[len(list(layers_names)) - config['no_of_layers_freeze']:]

        for name, param in base_network.named_parameters():
            if not name.replace('.weight', '').replace('.bias', '') in layers_freeze:
                param.requires_grad = False


    else:
        base_network = net_config["name"](**net_config["params"])
        base_network = base_network.cuda()
    parameter_list = base_network.get_parameters()

    ## set optimizer
    optimizer_config = config["optimizer"]
    optimizer = optimizer_config["type"](parameter_list, **(optimizer_config["optim_params"]))
    param_lr = []
    for param_group in optimizer.param_groups:
        param_lr.append(param_group["lr"])
    schedule_param = optimizer_config["lr_param"]
    lr_scheduler = lr_schedule.schedule_dict[optimizer_config["lr_type"]]

    gpus = config['gpu'].split(',')
    if len(gpus) > 1:
        base_network = nn.DataParallel(base_network, device_ids=[int(i) for i in gpus])

    return base_network, schedule_param, lr_scheduler, optimizer


def train(config, dset_loaders):
    # class_imb_weight = torch.FloatTensor(comepute_class_weight_pytorch()).cuda()

    logger = Logger(config["logs_path"] + "tensorboard/" + config['timestamp'])

    early_stopping = EarlyStopping(patience=config["patience"], verbose=True)

    base_network, schedule_param, lr_scheduler, optimizer = network_setup(config)

    ## train
    len_train_source = len(dset_loaders["source"])
    best_loss_valid = np.infty  # total

    for itr in range(config["num_iterations"]):
        if itr % config["snapshot_interval"] == 0:
            base_network.train(False)

            temp_model = nn.Sequential(base_network)
            torch.save(temp_model,
                       osp.join(config["model_path"],
                                "backup/model_iter_{:05d}.pth.tar".format(itr)))
        if itr % config["test_interval"] == 0:
            itr_log = "num_iterations  " + str(itr)
            config["out_file"].write(itr_log + "\n")
            config["out_file"].flush()
            # print(itr_log)
            base_network.train(False)
            val_info = validation_loss(dset_loaders, base_network, data_name ="valid_source",dset=config['dataset'],
                                       num_classes=config["network"]["params"]["class_num"],
                                       logs_path=config['logs_path'], num_iterations=itr,
                                       is_training=config['is_training'])

            print_msg("Iteration: " + str(itr) + "/"+ str(config["num_iterations"])+ " | Val loss: "+ str(val_info['val_loss'])+
                  " | Val Accuracy: "+ str(val_info['val_accuracy'])


                      ,config["out_file"])
            temp_model = nn.Sequential(base_network)
            if val_info['val_loss'] < best_loss_valid:
                # best_model = copy.deepcopy(temp_model)
                best_itr = itr
                best_loss_valid = val_info['val_loss']
                best_acc = val_info['val_accuracy']
                # best_cm  =    val_info['conf_mat']
                torch.save(temp_model, osp.join(config["model_path"], "best_model.pth.tar"))

                # torch.save(best_model, osp.join(config["model_path"], "model_iter_{:05d}_model.pth.tar".format(i)))

        early_stopping(val_info['val_loss'], nn.Sequential(base_network))
        if early_stopping.early_stop:
            print("Early stopping")
            print("Saving Model ...")

            break

        loss_params = config["loss"]
        ## train one iter
        base_network.train(True)
        optimizer = lr_scheduler(optimizer, itr, **schedule_param)
        optimizer.zero_grad()
        if itr % len_train_source == 0:
            iter_source = iter(dset_loaders["source"])

        inputs_source, labels_source,_ = iter_source.next()
        inputs_source, labels_source = inputs_source.cuda(), labels_source.cuda()

        features_source, outputs_source = base_network(inputs_source)


        classifier_loss = nn.CrossEntropyLoss()(outputs_source, labels_source)

        total_loss = classifier_loss

        total_loss.backward()
        optimizer.step()

        classifier_loss_numpy = classifier_loss.clone().cpu().detach().numpy()
        total_loss_numpy = total_loss.clone().cpu().detach().numpy()

        info = {'total_loss': total_loss_numpy.item(),
                'classifier_loss': classifier_loss_numpy.item(),

                'valid_source_loss': val_info['val_loss'], 'valid_source_acc': val_info['val_accuracy']
                }
        for tag, value in info.items():
            logger.scalar_summary(tag, value, itr)

        with open(config["logs_path"] + '/loss_values_.csv', mode='a') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            csv_writer.writerow(
                [itr, total_loss_numpy,  classifier_loss_numpy,                  val_info['val_loss'], val_info['val_accuracy'],

                 ])




    return config


def test(config, dset_loaders, model_path_for_testing=None):
    if model_path_for_testing:
        model = torch.load(model_path_for_testing)
    else:

        model = torch.load(osp.join(config["model_path"], "best_model.pth.tar"))


    val_info = validation_loss(dset_loaders, model, data_name='valid_source',dset=config['dataset'],
                               num_classes=config["network"]["params"]["class_num"],
                               logs_path=config['logs_path'], is_training=config['is_training'])


    test_info = validation_loss(dset_loaders, model, data_name='test',dset=config['dataset'],
                                num_classes=config["network"]["params"]["class_num"],
                                logs_path=config['logs_path'], is_training=config['is_training'])


    print_msg("Final Model " + "| Val loss: " +  str(val_info['val_loss']) + str( "| Val Accuracy: ") +str(
          val_info['val_accuracy'])+ ("| 2 class acc:"+str(val_info['val_acc_2_class']) if 'val_acc_2_class' in val_info else "") ,config["out_file"])
    print_msg("Final Model " + "| Test loss: " + str(test_info['val_loss']) + str("| Test Accuracy: ") +
              str(test_info['val_accuracy']) + (
                  "| 2 class acc:" + str(test_info['val_acc_2_class']) if 'val_acc_2_class' in test_info else ""),
              config["out_file"])

def print_msg(msg, outfile):
    print()
    print("=" * 50)
    print("" * 2, msg)
    print("=" * 50)
    print()

    outfile.write('\n')
    outfile.write("=" * 25)
    outfile.write(" " * 5 + msg)
    outfile.write("=" * 25)
    outfile.write('\n')
    outfile.flush()

def parge_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, choices=['train', 'test'])

    parser.add_argument('--seed', type=int)
    parser.add_argument('--dset', type=str, help="The dataset or source dataset used")
    parser.add_argument('--gpu_id', type=str, nargs='?', default='1', help="device id to run")

    parser.add_argument('--lr', type=float)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--power', type=float)
    parser.add_argument('--momentum', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--nesterov', type=float)
    parser.add_argument('--optimizer', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--batch_size_test', type=int)
    parser.add_argument('--use_bottleneck', type=bool)
    parser.add_argument('--bottleneck_dim', type=int)

    parser.add_argument('--new_cls', type=bool)
    parser.add_argument('--no_of_classes', type=int)
    parser.add_argument('--image_size', type=int)
    parser.add_argument('--crop_size', type=int)

    parser.add_argument('--num_iterations', type=int)
    parser.add_argument('--patience', type=int)
    parser.add_argument('--test_interval', type=int)
    parser.add_argument('--snapshot_interval', type=int)

    parser.add_argument('--trained_model_path', type=str)
    parser.add_argument('--no_of_layers_freeze', type=int)

    parser.add_argument('--s_dset', type=str)

    parser.add_argument('--test_dset_txt', type=str)
    parser.add_argument('--s_dset_txt', type=str)
    parser.add_argument('--sv_dset_txt', type=str)
    parser.add_argument('--output_dir', type=str)

    parser.set_defaults(
        mode="train",
        seed=0,
        gpu_id="0",
        dset="malaria",
        s_dset_txt='data/malaria/MD4/txt_70_20_10/MD4_random_train.txt',
        sv_dset_txt='data/malaria/MD4/txt_70_20_10/MD4_random_valid.txt',
        test_dset_txt='data/malaria/MD4/txt_70_20_10/MD4_random_test.txt',

        s_dset="MD4",

        lr=0.01,
        arch="ResNet50",
        gamma=0.0001,
        power=0.75,
        momentum=0.9,
        weight_decay=0.0005,
        nesterov=True,
        optimizer="SGD",
        batch_size=2,
        batch_size_test=128,
        use_bottleneck=False,
        bottleneck_dim=256,
        new_cls=True,
        no_of_classes=2,
        image_size=256,
        crop_size=224,
        trained_model_path= None,
        no_of_layers_freeze=13,

        num_iterations=2,
        patience=2000,
        test_interval=50,
        snapshot_interval=1000,
        output_dir="experiments"
    )

    args = parser.parse_args()
    return args


def set_deterministic_settings(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    ####################################
    # Default Project Folders#
    ####################################

    project_root = "../../"
    data_root = project_root + "data/"
    models_root = project_root + "models/"

    now = datetime.now()
    timestamp = now.strftime("%d/%m/%Y %H:%M:%S")
    timestamp = timestamp.replace("/", "_").replace(" ", "_").replace(":", "_").replace(".", "_")

    args = parge_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
    set_deterministic_settings(seed=args.seed)

    dataset = args.dset

    log_output_dir_root = args.output_dir + '/logs/' + dataset + '/'
    models_output_dir_root = args.output_dir + '/models/' + dataset + '/'

    # print(os.listdir(project_root))
    if args.mode == "train":
        is_training = True
    else:
        is_training = False

    config = {}
    no_of_classes = args.no_of_classes

    trial_number = args.mode + "_" + args.s_dset + "_CNN"

    ####################################
    # Dataset Locations Setup #
    ####################################
    source_input = {'path': args.s_dset_txt}
    source_valid_input = {'path': args.sv_dset_txt}
    test_input = {'path': args.test_dset_txt, 'labelled': True}


    if not is_training:
        model_path_for_testing = models_root + args.dset + '/train_' + args.s_dset +"_CNN" + "/best_model.pth.tar"

        if args.trained_model_path:
            model_path_for_testing = args.trained_model_path

    config['timestamp'] = timestamp
    config['trial_number'] = trial_number
    config["gpu"] = args.gpu_id
    config["num_iterations"] = args.num_iterations
    config["test_interval"] = args.test_interval
    config["snapshot_interval"] = args.snapshot_interval
    config["patience"] = args.patience
    config["is_training"] = is_training

    if not is_training:
        config["num_iterations"] = 0
        best_itr = "testing"
        print("Testing:")
        config["best_itr"] = "testing"

    print("num_iterations", config["num_iterations"])

    log_output_path = log_output_dir_root + args.arch + '/' +  trial_number + '/'
    trial_results_path = models_output_dir_root + args.arch + '/' + trial_number + '/'
    config["model_path"] = trial_results_path
    config["logs_path"] = log_output_path
    if not os.path.exists(config["logs_path"]):
        os.makedirs(config["logs_path"])

    if is_training:
        if not os.path.exists(config["model_path"] + "/backup/"):
            os.makedirs(config["model_path"] + "/backup/")

    config["out_file"] = open(osp.join(config["logs_path"], "log.txt"), "w")
    resize_size = args.image_size

    config["prep"] = {'params_source': {"resize_size": resize_size, "crop_size": args.crop_size, "dset": dataset},
                      'params_test': {"resize_size": resize_size, "crop_size": args.crop_size, "dset": dataset}}

    config["loss"] = {"trade_off": 1.0}
    config["trained_model_path"] = args.trained_model_path
    config['no_of_layers_freeze'] = args.no_of_layers_freeze

    if "Xception" in args.arch:
        config["network"] = \
            {"name": network.XceptionFc,
             "params":
                 {
                     "use_bottleneck": args.use_bottleneck,
                     "bottleneck_dim": args.bottleneck_dim,
                     "new_cls": args.new_cls}}
    elif "ResNet50" in args.arch:
        config["network"] = {"name": network.ResNetFc,
                             "params":
                                 {"resnet_name": args.arch,
                                  "use_bottleneck": args.use_bottleneck,
                                  "bottleneck_dim": args.bottleneck_dim,
                                  "new_cls": args.new_cls}}

    elif "Inception" in args.arch:
        config["network"] = {"name": network.Inception3Fc,
                             "params":
                                 {"use_bottleneck": args.use_bottleneck,
                                  "bottleneck_dim": args.bottleneck_dim,
                                  "new_cls": args.new_cls}}

    if args.optimizer == "SGD":

        config["optimizer"] = {"type": optim.SGD, "optim_params": {'lr': args.lr, "momentum": args.momentum,
                                                                   "weight_decay": args.weight_decay,
                                                                   "nesterov": args.nesterov},
                               "lr_type": "inv",
                               "lr_param": {"lr": args.lr, "gamma": args.gamma, "power": args.power}}

    elif args.optimizer == "Adam":
        config["optimizer"] = {"type": optim.Adam, "optim_params": {'lr': args.lr,
                                                                    "weight_decay": args.weight_decay},
                               "lr_type": "inv",
                               "lr_param": {"lr": args.lr, "gamma": args.gamma, "power": args.power}}

    config["dataset"] = dataset
    config["data"] = {"source": {"list_path": source_input['path'], "batch_size": args.batch_size},
                      "test": {"list_path": test_input['path'], "batch_size": args.batch_size_test,
                               "labelled": test_input['labelled']},
                      "valid_source": {"list_path": source_valid_input['path'], "batch_size": args.batch_size}}
    config["optimizer"]["lr_param"]["lr"] = args.lr
    config["network"]["params"]["class_num"] = no_of_classes

    config["out_file"].write(str(config))
    config["out_file"].flush()
    print("source_path", source_input)
    print("test_path", test_input)
    # print('GPU', os.environ["CUDA_VISIBLE_DEVICES"], config["gpu"])

    ####################################
    # Dump arguments #
    ####################################
    with open(config["logs_path"] + "args.yml", "w") as f:
        yaml.dump(args, f)

    dset_loaders = data_setup(config)

    if is_training:
        print()
        print("=" * 50)
        print(" " * 15, "Training Started")
        print("=" * 50)
        print()

        train(config, dset_loaders)
        print()
        print("=" * 50)
        print(" " * 15, "Testing Started")
        print("=" * 50)
        print()
        test(config, dset_loaders)
    else:
        print()
        print("=" * 50)
        print(" " * 15, "Testing Started")
        print("=" * 50)
        print()

        test(config, dset_loaders, model_path_for_testing=model_path_for_testing)


if __name__ == "__main__":
    main()
