import csv
import random
import statistics

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix


# def comepute_class_weight_pytorch():
#     dat_list = open(config["data"]["source"]["list_path"]).readlines()
#     lables = [int(line.split(" ")[1]) for line in dat_list]
#
#     y_c = list(np.unique(lables))
#
#     return compute_class_weight('balanced', y_c, lables)

def testing_sperm_slides(loader, model, logs_path, slide_annotation_file_path, num_classes, print_info=True):
    print()
    print("|| Testing on patient slides ||")
    print()

    start_test = True
    with torch.no_grad():
        iter_test = iter(loader["test"])
        for i in range(len(loader['test'])):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            labels = labels
            # _, outputs = model(inputs)
            _, predict_out = model(inputs)
            outputs = nn.Softmax(dim=1)(predict_out)
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                # all_label = torch.cat((all_label, labels.float()), 0)
                all_label = all_label + labels

    all_output_numpy = all_output.numpy()

    _, predict = torch.max(all_output, 1)

    all_values_CSV = []

    predict_numpy = predict.numpy()



    with open(logs_path + '/target_predict_conf.csv', mode='w') as file:
        csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for value in range(len(all_output_numpy)):
            # csv_writer.writerow([all_label[value], all_output_numpy[value][0], all_output_numpy[value][1]])

            csv_writer.writerow(['Image_Name', 'Prediction'] + list(map(lambda x: 'class_' + str(x) + '_conf',
                                                                        np.arange(
                                                                            num_classes))))  # ['Image_Name', 'Prediction', 'class_0_conf', 'class_1_conf']

            for value in range(len(all_output_numpy)):
                csv_writer.writerow(
                    [all_label[value], predict_numpy[value]] +
                    list(map(lambda x: all_output_numpy[value][x], np.arange(num_classes))))




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

        true_value_slide = 'Not_Found'
        # with open('../../data/sd1_sd2_sd3_patient_doctor.csv') as csv_file:
        with open(slide_annotation_file_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')

            for row in csv_reader:

                if row[0] == key:
                    # print(key, row[0])

                    true_value_slide = row[1]

        main_image_results.append(
            [key, zero_per, one_per, count0, count1, total, true_value_slide, sd, mean])


        if print_info:
            print("|",key+":-", "| good_sperm ", round(zero_per, 3), '%  |  Counts: ' + str(count0) + '/' + str(total), '|  Clinicians value: ',true_value_slide,"%  |")
    with open(logs_path + '/slide_prediction_sperm.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(["Slide_name", "Good Sperm Percentage", "Bad Sperm Percentage", "Good sperm count",
                         "Bad sperm count", "Total Count", "Clinician's annotation"," Standard deviation Good Sperm Percentage", "Mean Good Sperm Percentage"])
        writer.writerows(main_image_results)

    f.close()

def class_5_to2(all_label_numpy,predict_numpy):
    preds_bnb_logits = []
    y_real_bnb_logits = []
    for i in range(len(all_label_numpy)):
        # print(all_label_numpy)
        if (all_label_numpy[i] < 2):
            y_real_bnb_each = 0
        else:
            y_real_bnb_each = 1
        y_real_bnb_logits.append(y_real_bnb_each)

        if (predict_numpy[i] < 2):

            y_pred_bnb_each = 0
        else:
            y_pred_bnb_each = 1

        preds_bnb_logits.append(y_pred_bnb_each)
    # preds_bnb_logits = np.array(preds_bnb_logits)
    # y_real_bnb_logits = np.array(y_real_bnb_logits)
    # y_real_bnb = np.argmax(y_real_bnb_logits, axis=-1)
    # preds_bnb = np.argmax(preds_bnb_logits, axis=-1)
    preds_bnb = preds_bnb_logits
    # cm_bnb = confusion_matrix(y_real_bnb_logits, preds_bnb)
    # print(preds_bnb)
    cm_bnb = confusion_matrix(y_true=y_real_bnb_logits, y_pred=preds_bnb)
    print(cm_bnb)
    acc = 0.
    sum_ = 0.
    for i in range(cm_bnb.shape[0]):
        for j in range(cm_bnb.shape[1]):
            if i == j:
                acc += cm_bnb[i][j]
            sum_ += cm_bnb[i][j]

    acc = acc * 100 / sum_
    print("2 class Accuracy = ", acc, "%")

    return cm_bnb, acc

def validation_loss(loader, model, num_classes, logs_path, data_name='valid_source', dset="",num_iterations=0, is_training=True,
                    ret_cm=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader[data_name])
        for i in range(len(loader[data_name])):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            paths = data[2]
            inputs = inputs.cuda()
            # labels = labels.cuda()
            _, raw_outputs = model(inputs)
            outputs = nn.Softmax(dim=1)(raw_outputs)
            if start_test:
                all_output = outputs.cpu()
                all_label = labels
                # all_path= paths.str()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.cpu()), 0)
                all_label = torch.cat((all_label, labels), 0)
                # all_path = all_path + paths
    val_loss = nn.CrossEntropyLoss()(all_output, all_label)

    val_loss = val_loss.numpy().item()

    all_output = all_output.float()
    _, predict = torch.max(all_output, 1)

    all_label = all_label.float()
    val_accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])

    all_output_numpy = all_output.numpy()
    predict_numpy = predict.numpy()

    with open(logs_path + '/'+dset+"_" + data_name + "_"+ (str(num_iterations) if is_training else "Final") + '_confidence_values_.csv',
              mode='w') as file:
        csv_writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        csv_writer.writerow(['Image_Name', 'Label', 'Prediction'] + list(map(lambda x: 'class_' + str(x) + '_conf', np.arange(
            num_classes))))  # ['Image_Name', 'Prediction', 'class_0_conf', 'class_1_conf']

        for value in range(len(all_output_numpy)):
            csv_writer.writerow(
                [
                    # all_path[value],
                 int(all_label[value].item()), predict_numpy[value]] +
                list(map(lambda x: all_output_numpy[value][x], np.arange(num_classes))))

    conf_mat = confusion_matrix(all_label, torch.squeeze(predict).float())
    val_info = {"val_accuracy": val_accuracy, "val_loss": val_loss}

    if num_classes == 5:
        cm_bnb, acc_2_class = class_5_to2(all_label_numpy=all_label.numpy(), predict_numpy=predict_numpy)

        val_info = {**val_info, "cm_bnb": cm_bnb, "val_acc_2_class": acc_2_class}
    elif num_classes == 2:

        val_info = {**val_info,  "val_acc_2_class": val_accuracy}


    if ret_cm:
        val_info = {**val_info, "conf_mat": conf_mat}
    return val_info

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1

def calc_transfer_loss(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target)

def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)



def obtain_label(loader, model,num_classes):
    epsilon = 1e-5
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            feas,outputs = model(inputs)
            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + epsilon), dim=1)
    unknown_weight = 1 - ent / np.log(num_classes)
    _, predict = torch.max(all_output, 1)

    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # if args.distance == 'cosine':
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > 0)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], 'cosine')
    # print(dd)
    pred_label = dd.argmin(axis=1)
    # print(pred_label)

    pred_label = labelset[pred_label]
    # print(pred_label)

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]
    # print(pred_label)

    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)


    print(log_str + '\n')

    return torch.from_numpy(pred_label.astype('int')).cuda()
def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer

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