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
import helper_utils.pre_process as prep
import yaml
from scipy.spatial.distance import cdist

from torch.utils.data import DataLoader
import helper_utils.lr_schedule as lr_schedule
from helper_utils.data_list_m import ImageList_idx as ImageList

import argparse

from helper_utils.logger import Logger
from helper_utils.sampler import ImbalancedDatasetSampler

from helper_utils.EarlyStopping import EarlyStopping
from helper_utils.tools import testing_sperm_slides, validation_loss, calc_transfer_loss, Entropy,op_copy,obtain_label,lr_scheduler
from helper_utils.tools import op_copy,obtain_label,lr_scheduler


def data_setup(config):
    ## set pre-process
    prep_dict = {}
    prep_config = config["prep"]
    prep_dict["target"] = prep.image_train_nos(**config["prep"]['params_target'])
    prep_dict["test"] = prep.image_test_nos(**config["prep"]['params_target'])
    prep_dict["target_valid"] = prep.image_test_nos(**config["prep"]['params_target'])

    ## prepare data
    dsets = {}
    dset_loaders = {}
    data_config = config["data"]
    train_bs = data_config["target"]["batch_size"]
    test_bs = data_config["test"]["batch_size"]


    dsets["target"] = ImageList(open(data_config["target"]["list_path"]).readlines(),
                                transform=prep_dict["target"])
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs,
                                        shuffle=True, num_workers=4, drop_last=True)

    dsets["target_valid"] = ImageList(open(data_config["target_valid"]["list_path"]).readlines(),
                                      transform=prep_dict["target_valid"],
                                      )
    dset_loaders["target_valid"] = DataLoader(dsets["target_valid"], batch_size=test_bs,
                                              shuffle=False, num_workers=4)

    dsets["test"] = ImageList(open(data_config["test"]["list_path"]).readlines(),
                              transform=prep_dict["test"])
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=test_bs, \
                                      shuffle=False, num_workers=4)

    return dset_loaders
def print_msg(msg, outfile):
    print()
    print("=" * 50)
    print(" " * 15, msg)
    print("=" * 50)
    print()

    outfile.write('\n')
    outfile.write("=" * 25)
    outfile.write(" " * 5 + msg)
    outfile.write("=" * 25)
    outfile.write('\n')
    outfile.flush()


def network_setup(config):
    class_num = config["network"]["params"]["class_num"]

    ## set base network
    net_config = config["network"]

    base_network = net_config["name"](**net_config["params"])
    base_network_source = torch.load(config['source_model_path'])[0]
    base_network.load_state_dict(base_network_source.state_dict())


    base_network = base_network.cuda()
    base_network_source =  base_network_source.cuda()

    ad_net = network.AdversarialNetwork(base_network.output_num() * class_num, 1024)
    ad_net = ad_net.cuda()

    learning_rate = config["optimizer"]["optim_params"]["lr"]

    param_group = [
                    {'params': base_network.feature_layers.parameters(), 'lr': learning_rate * 0.1},
                    {'params': base_network.bottleneck.parameters(), 'lr': learning_rate },
                    {'params': base_network.fc.parameters(), 'lr': learning_rate },
                    {'params': base_network.bn.parameters(), 'lr': learning_rate }

                   ]
    for pg in param_group:
        pg['weight_decay'] = config["optimizer"]["optim_params"]["weight_decay"]
        pg['momentum'] = config["optimizer"]["optim_params"]["momentum"]
        pg['nesterov'] = config["optimizer"]["optim_params"]["nesterov"]
    optimizer = config["optimizer"]["type"](param_group)
    optimizer = op_copy(optimizer)


    return base_network,base_network_source, ad_net, optimizer


def train(config, dset_loaders):
    # class_imb_weight = torch.FloatTensor(comepute_class_weight_pytorch()).cuda()
    class_num = config["network"]["params"]["class_num"]

    def one_hot(pos):
        a = np.zeros(class_num)
        a[pos] = 1.0
        return a

    logger = Logger(config["logs_path"] + "tensorboard/" + config['timestamp'])

    early_stopping = EarlyStopping(patience=config["patience"], verbose=True)

    base_network,base_network_source, ad_net, optimizer = network_setup(config)

    ## train
    len_train_target = len(dset_loaders["target"])

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
            val_info = validation_loss(dset_loaders, base_network, data_name ="target_valid",dset=config['dataset'],
                                       num_classes=config["network"]["params"]["class_num"],
                                       logs_path=config['logs_path'], num_iterations=itr,
                                       is_training=config['is_training'])

            print_msg("Iteration: " + str(itr) + "/"+ str(config["num_iterations"])+ " | Val loss: "+ str(val_info['val_loss'])+
                  " | Val Accuracy: "+ str(val_info['val_accuracy'])


                      ,config["out_file"])


            pseudo_label_target = obtain_label(dset_loaders['test'], model=base_network, num_classes=class_num)

            temp_model = nn.Sequential(base_network)
            if val_info['val_loss'] < best_loss_valid:
                best_itr = itr
                best_loss_valid = val_info['val_loss']
                best_acc = val_info['val_accuracy']
                # best_cm  =    val_info['conf_mat']

                # torch.save(best_model, osp.join(config["model_path"], "model_iter_{:05d}_model.pth.tar".format(i)))
                torch.save(temp_model, osp.join(config["model_path"], "best_model.pth.tar"))



        early_stopping(val_info['val_loss'], nn.Sequential(base_network))

        if early_stopping.early_stop:
            print("Early stopping")
            print("Saving Model ...")

            break

        optimizer = lr_scheduler(optimizer, iter_num=itr, max_iter=config["num_iterations"])

        loss_params = config["loss"]
        ## train one iter
        base_network.train(True)
        base_network_source.train(False)
        ad_net.train(True)
        # optimizer = lr_scheduler(optimizer, itr, **schedule_param)
        lr_scheduler(optimizer, iter_num=itr, max_iter=config["num_iterations"])

        optimizer.zero_grad()

        if itr % len_train_target == 0:
            iter_target = iter(dset_loaders["target"])

        inputs_target, _, idxs = next(iter_target)

        inputs_target = inputs_target.cuda()

        features_source, outputs_source = base_network_source(inputs_target)  # change to target path

        features_target, outputs_target = base_network(inputs_target)

        features = torch.cat((features_source, features_target), dim=0)

        softmax_out_source = nn.Softmax(dim=1)(outputs_source)

        softmax_out_t = pseudo_label_target[idxs]
        softmax_out_target_float = torch.from_numpy(
            np.array([one_hot(int(i)) for i in softmax_out_t])).float().cuda()
        softmax_out = torch.cat((softmax_out_source, softmax_out_target_float), dim=0)

        classifier_loss_shot = nn.CrossEntropyLoss()(outputs_target, softmax_out_t)

        entropy = Entropy(softmax_out)
        transfer_loss = calc_transfer_loss([features, softmax_out], ad_net, entropy, network.calc_coeff(itr))

        total_loss = transfer_loss + loss_params["trade_off_cls"] * classifier_loss_shot

        total_loss.backward()
        optimizer.step()

        ####################################
        # Save iteration logs#
        ####################################

        transfer_loss_numpy = transfer_loss.clone().cpu().detach().numpy()
        classifier_loss_numpy = classifier_loss_shot.clone().cpu().detach().numpy()
        total_loss_numpy = total_loss.clone().cpu().detach().numpy()
        entropy_numpy = torch.sum(entropy).clone().cpu().detach().numpy()

        # info = {'total_loss': total_loss_numpy.item(),
        #         'classifier_loss': classifier_loss_numpy.item(), 'transfer_loss': transfer_loss_numpy.item(),
        #         'entropy': entropy_numpy.item(),
        #
        #         'valid_source_loss': val_info['val_loss'], 'valid_source_acc': val_info['val_accuracy']
        #         }
        # for tag, value in info.items():
        #     logger.scalar_summary(tag, value, itr)

        with open(config["logs_path"] + '/loss_values_.csv', mode='a') as file:
            csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            csv_writer.writerow(
                [itr, total_loss_numpy, transfer_loss_numpy, classifier_loss_numpy, entropy_numpy,
                 val_info['val_loss'], val_info['val_accuracy'],

                 ])

    return config


def test(config, dset_loaders, model_path_for_testing=None):
    config['is_training']= False
    if model_path_for_testing:
        model = torch.load(model_path_for_testing)
    else:

        model = torch.load(osp.join(config["model_path"], "best_model.pth.tar"))

    val_info = validation_loss(dset_loaders, model, data_name='target_valid',dset=config['dataset'],
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
    parser.add_argument('--source_model_path', type=str)


    parser.add_argument('--no_of_layers_freeze', type=int)

    parser.add_argument('--s_dset', type=str)
    parser.add_argument('--t_dset', type=str)

    parser.add_argument('--test_dset_txt', type=str)

    parser.add_argument('--tv_dset_txt', type=str)
    parser.add_argument('--t_dset_txt', type=str)
    parser.add_argument('--target_labelled', type=str, default='True', choices=['True', 'False', 'true', 'false'], )
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--sperm_patient_data_clinicians_annotations', type=str)
    parser.add_argument('--trade_off_cls', type=float)

    parser.set_defaults(
        mode="train",
        seed=0,
        gpu_id="0",
        dset="embryo",
        tv_dset_txt='../../data/office/txt_files/W10.txt',
        t_dset_txt='../../data/office/txt_files/W.txt',
        source_model_path="../submitted_models/office/source_only/A_best_model.pth.tar",

        s_dset="A",
        t_dset="W",

        lr=0.0001,
        arch="ResNet50",
        gamma=10,
        power=0.75,
        momentum=0.9,
        weight_decay=1e-3,
        nesterov=True,
        optimizer="SGD",
        batch_size=32,
        batch_size_test=256,
        use_bottleneck=True,
        bottleneck_dim=256,
        new_cls=True,
        no_of_classes=31,
        image_size=256,
        crop_size=224,
        trained_model_path= None,
        num_iterations=20,
        patience=5000,
        test_interval=10,
        snapshot_interval=20000,
        target_labelled="True",
        output_dir="../../",
        trade_off_cls=0.7,
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

    trial_number = args.mode + "_" + args.s_dset + "_to_" + args.t_dset

    ####################################
    # Dataset Locations Setup #
    ####################################

    target_valid_input = {'path': args.tv_dset_txt}
    target_input = {'path': args.t_dset_txt, 'labelled': args.target_labelled.lower() == 'true'}
    if args.test_dset_txt:
        test_input = {'path': args.test_dset_txt, 'labelled': args.target_labelled.lower() == 'true'}
    else:
        test_input = target_input

    if not is_training:
        model_path_for_testing = models_root + args.dset + '/' + args.s_dset + "=>" + args.t_dset + "/best_model.pth.tar"

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
    config['source_model_path'] = args.source_model_path
    if not is_training:
        config["num_iterations"] = 0
        best_itr = "testing"
        print("Testing:")
        config["best_itr"] = "testing"

    print("num_iterations", config["num_iterations"])

    log_output_path = log_output_dir_root + args.arch + '/' + trial_number + '/'
    trial_results_path = models_output_dir_root + args.arch+"/" + trial_number + '/'
    config["model_path"] = trial_results_path
    config["logs_path"] = log_output_path
    if not os.path.exists(config["logs_path"]):
        os.makedirs(config["logs_path"])

    if is_training:
        if not os.path.exists(config["model_path"] + "/backup/"):
            os.makedirs(config["model_path"] + "/backup/")

    config["out_file"] = open(osp.join(config["logs_path"], "log.txt"), "w")
    resize_size = args.image_size

    config["prep"] = {
                      'params_target': {"resize_size": resize_size, "crop_size": args.crop_size, "dset": dataset}}

    config["loss"] = {"trade_off_cls": args.trade_off_cls}
    config["trained_model_path"] = args.trained_model_path

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
    config["data"] = {
                      "target": {"list_path": target_input['path'], "batch_size": args.batch_size,
                                 "labelled": target_input['labelled']},
                      "test": {"list_path": test_input['path'], "batch_size": args.batch_size_test,
                               "labelled": test_input['labelled']},
                      "target_valid": {"list_path": target_valid_input['path'], "batch_size": args.batch_size}}
    config["optimizer"]["lr_param"]["lr"] = args.lr
    config["network"]["params"]["class_num"] = no_of_classes

    config["out_file"].write(str(config))
    config["out_file"].flush()
    print("target_path", target_input)
    print("target_valid_input", target_valid_input)
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
