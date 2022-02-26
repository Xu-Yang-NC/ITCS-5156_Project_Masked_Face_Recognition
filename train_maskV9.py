import torch
torch.cuda.is_available()
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

sys.path.append(os.getcwd())

from torch.nn.modules.distance import PairwiseDistance
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import time

from Data_loader.Data_loader_facenet_mask import test_dataloader,V9_train_dataloader
from Data_loader.Data_loader_facenet_mask import LFWestMask_dataloader
from Losses.Triplet_loss import TripletLoss
from validate_on_LFW import evaluate_lfw
from Data_loader.Data_loader_train_notmask import TrainDataset

from config_mask import config
from Models.CBAM_Face_attention_Resnet_notmaskV3 import resnet18_cbam, resnet50_cbam, resnet101_cbam, resnet34_cbam, \
    resnet152_cbam
pwd = os.path.abspath('./')
print(torch.cuda.is_available())
print("Using {} model architecture.".format(config['model']))
start_epoch = 0

# Model 34 is picked for this training
if config['model'] == 18:
    model = resnet18_cbam(pretrained=True, showlayer= False,num_classes=128)
elif config['model'] == 34:
    model = resnet34_cbam(pretrained=True, showlayer= False, num_classes=128)
elif config['model'] == 50:
    model = resnet50_cbam(pretrained=True, showlayer= False, num_classes=128)
elif config['model'] == 101:
    model = resnet101_cbam(pretrained=True, showlayer= False, num_classes=128)
elif config['model'] == 152:
    model = resnet152_cbam(pretrained=True, showlayer= False, num_classes=128)

model_path = os.path.join(pwd, 'Model_training_checkpoints')
x = [int(i.split('_')[4]) for i in os.listdir(model_path) if 'V9' in i]
x.sort()
for i in os.listdir(model_path):
    if (len(x)!=0) and ('epoch_'+str(x[-1]) in i) and ('V9' in i):
        model_pathi = os.path.join(model_path, i)
        break

if os.path.exists(model_pathi) and ('V9' in model_pathi):
    model_state = torch.load(model_pathi)
    model.load_state_dict(model_state['model_state_dict'])
    start_epoch = model_state['epoch']
    print('loaded %s' % model_pathi)
else:
    print('Training dataset is not exist！')

    

flag_train_gpu = torch.cuda.is_available()
flag_train_multi_gpu = False
if flag_train_gpu and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    model.cuda()
    flag_train_multi_gpu = True
    print('Using multi-gpu training.')
elif flag_train_gpu and torch.cuda.device_count() == 1:
    model.cuda()
    print('Using single-gpu training.')

def adjust_learning_rate(optimizer, epoch):
    if epoch<19:
        lr =  0.125
    elif (epoch>=19) and (epoch<60):
        lr = 0.0625
    elif (epoch >= 60) and (epoch < 90):
        lr = 0.0155
    elif (epoch >= 90) and (epoch < 120):
        lr = 0.003
    elif (epoch>=120) and (epoch<160):
        lr = 0.0001
    else:
        lr = 0.00006
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Adam is picked as the optimizer
def create_optimizer(model, new_lr):
    # setup optimizer
    if config['optimizer'] == "sgd":
        optimizer_model = torch.optim.SGD(model.parameters(), lr = new_lr,
                              momentum=0.9, dampening=0.9,
                              weight_decay=0)
    elif config['optimizer'] == "adagrad":
        optimizer_model = torch.optim.Adagrad(model.parameters(), lr = new_lr,
                                  lr_decay=1e-4,
                                  weight_decay=0)

    elif config['optimizer'] == "rmsprop":
        optimizer_model = torch.optim.RMSprop(model.parameters(), lr = new_lr)

    elif config['optimizer'] == "adam":
        optimizer_model = torch.optim.Adam(model.parameters(), lr = new_lr,
                               weight_decay=0)
    return optimizer_model

# Random seed
seed = 0
optimizer_model = create_optimizer(model, 0.125)
torch.manual_seed(seed)  # Set seed for CPU
torch.cuda.manual_seed(seed)  # Set seed for current GPU
torch.cuda.manual_seed_all(seed)  # Set seed for all the GPU
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

# epoch
total_time_start = time.time()
start_epoch = start_epoch
end_epoch = start_epoch + config['epochs']
# import l2 calculation
l2_distance = PairwiseDistance(2).cuda()
# For report
best_roc_auc = -1
best_accuracy = -1
print('Countdown 3 seconds')
time.sleep(1)
print('Countdown 2 seconds')
time.sleep(1)
print('Countdown 1 seconds')
time.sleep(1)

# epoch loop
for epoch in range(start_epoch, end_epoch):
    torch.cuda.empty_cache()
    print("\ntraining on TrainDataset! ...")
    epoch_time_start = time.time()
    triplet_loss_sum = 0
    attention_loss_sum = 0
    num_hard = 0

    model.train()  # train the model
    # step loop
    progress_bar = enumerate(tqdm(V9_train_dataloader))
    for batch_idx, (batch_sample) in progress_bar:
        # for batch_idx, (batch_sample) in enumerate(train_dataloader):
        # length = len(train_dataloader)
        # fl=open('/home/Mask-face-recognitionV1/output.txt', 'w')
        # for batch_idx, (batch_sample) in enumerate(train_dataloader):
        # print(batch_idx, end=' ')
        # fl.write(str(batch_idx)+' '+str(round((time.time()-epoch_time_start)*length/((batch_idx+1)*60), 2))+'；  ')
        # Getting the datasets for the iteration
        # Getting three pictures without mask
        anc_img = batch_sample['anc_img'].cuda()
        pos_img = batch_sample['pos_img'].cuda()
        neg_img = batch_sample['neg_img'].cuda()
        # Getting three pictures with mask
        mask_anc = batch_sample['mask_anc'].cuda()
        mask_pos = batch_sample['mask_pos'].cuda()
        mask_neg = batch_sample['mask_neg'].cuda()

        # model calculation
        #  Feedforward- the model is trained by three pictures, embedding and loss is generated. (only two pictures are input with loss during training, and one picture is input with embedding during testing)
        anc_embedding, anc_attention_loss = model((anc_img, mask_anc))
        pos_embedding, pos_attention_loss = model((pos_img, mask_pos))
        neg_embedding, neg_attention_loss = model((neg_img, mask_neg))
        anc_embedding = torch.div(anc_embedding, torch.norm(anc_embedding)) * 50
        pos_embedding = torch.div(pos_embedding, torch.norm(pos_embedding)) * 50
        neg_embedding = torch.div(neg_embedding, torch.norm(neg_embedding)) * 50
        
        # Calculating l2 embedding
        pos_dist = l2_distance.forward(anc_embedding, pos_embedding)
        neg_dist = l2_distance.forward(anc_embedding, neg_embedding)
        # Looking for difficult sample
        all = (neg_dist - pos_dist < config['margin']).cpu().numpy().flatten()
        hard_triplets = np.where(all == 1)
        if len(hard_triplets[0]) == 0:
            continue

        # Select embedding of the difficult samples
        anc_hard_embedding = anc_embedding[hard_triplets].cuda()
        pos_hard_embedding = pos_embedding[hard_triplets].cuda()
        neg_hard_embedding = neg_embedding[hard_triplets].cuda()
        # Select attention loss of the difficult samples
        hard_anc_attention_loss = anc_attention_loss[hard_triplets]
        hard_pos_attention_loss = pos_attention_loss[hard_triplets]
        hard_neg_attention_loss = neg_attention_loss[hard_triplets]

        # loss calculation
        triplet_loss = TripletLoss(margin=config['margin']).forward(
            anchor=anc_hard_embedding,
            positive=pos_hard_embedding,
            negative=neg_hard_embedding
        ).cuda()

        # Calculating the mean for attention loss
        hard_attention_loss = torch.cat([hard_anc_attention_loss, hard_pos_attention_loss, hard_neg_attention_loss])
        # hard_attention_loss = torch.cat([anc_attention_loss, pos_attention_loss, neg_attention_loss])
        hard_attention_loss = torch.mean(hard_attention_loss).cuda()
        hard_attention_loss = hard_attention_loss.type(torch.FloatTensor)
        
        LOSS = triplet_loss + hard_attention_loss

        # Back propagation process
        optimizer_model.zero_grad()
        LOSS.backward()
        optimizer_model.step()

        # update the optimizer learning rate
        adjust_learning_rate(optimizer_model, epoch)

        # Record log info
        # Record the # of difficult sample
        num_hard += len(anc_hard_embedding)
        # Calculate the total loss in this epoch
        triplet_loss_sum += triplet_loss.item()
        attention_loss_sum += hard_attention_loss.item()
       

    # Calculate the average loss in this epoch
    avg_triplet_loss = 0 if (num_hard == 0) else triplet_loss_sum / num_hard
    avg_attention_loss = 0 if (num_hard == 0) else attention_loss_sum / num_hard
    avg_loss = avg_triplet_loss + avg_attention_loss
    epoch_time_end = time.time()

    # Calculate the testing accuracy
    print("Validating on TestDataset! ...")
    model.eval()  # Model evaluation
    with torch.no_grad():
        distances, labels = [], []

        progress_bar = enumerate(tqdm(test_dataloader))
        for batch_index, (data_a, data_b, label) in progress_bar:
            # data_a, data_b, label are in one matrix
            data_a = data_a.cuda()
            data_b = data_b.cuda()
            label = label.cuda()
            output_a, output_b = model(data_a), model(data_b)
            output_a = torch.div(output_a, torch.norm(output_a))
            output_b = torch.div(output_b, torch.norm(output_b))
            distance = l2_distance.forward(output_a, output_b)
            # Matrices in a list
            labels.append(label.cpu().detach().numpy())
            distances.append(distance.cpu().detach().numpy())
        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for distance in distances for subdist in distance])
        true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
        tar, far = evaluate_lfw(
            distances=distances,
            labels=labels,
            epoch = 'epoch_'+str(epoch),
            tag = 'NOTMaskedLFW_auc',
            version = 'V9',
            pltshow=True
        )
    print("Validating on LFWMASKTestDataset! ...")
    with torch.no_grad():  
        distances, labels = [], []
        progress_bar = enumerate(tqdm(LFWestMask_dataloader))
        for batch_index, (data_a, data_b, label) in progress_bar:
            # data_a, data_b, label are in one matrix
            data_a = data_a.cuda()
            data_b = data_b.cuda()
            label = label.cuda()
            output_a, output_b = model(data_a), model(data_b)
            output_a = torch.div(output_a, torch.norm(output_a))
            output_b = torch.div(output_b, torch.norm(output_b))
            distance = l2_distance.forward(output_a, output_b)
            # Matrices in a list
            labels.append(label.cpu().detach().numpy())
            distances.append(distance.cpu().detach().numpy())
        
        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for distance in distances for subdist in distance])
        true_positive_rate_mask, false_positive_rate_mask, precision_mask, recall_mask, \
        accuracy_mask, roc_auc_mask, best_distances_mask, \
        tar_mask, far_mask = evaluate_lfw(
            distances=distances,
            labels=labels,
            epoch = 'epoch_'+str(epoch),
            tag = 'MaskedLFW_auc',
            version = 'V9',
            pltshow=True
        )

    # Print and save report
    # Read the best roc and acc value from previous file and update them
    if os.path.exists('logs/lfw_{}_log_tripletnotmaskV9.txt'.format(config['model'])):
        with open('logs/lfw_{}_log_tripletnotmaskV9.txt'.format(config['model']), 'r') as f:
            lines = f.readlines()
            my_line = lines[-3]
            my_line = my_line.split('\t')
            best_roc_auc = float(my_line[3].split(':')[1])
            best_accuracy = float(my_line[5].split(':')[1])

    # Save when the last epoch and auc is the higest
    save = True
    if config['save_last_model'] and epoch == end_epoch - 1:
        save = True
    if roc_auc > best_roc_auc:
        best_roc_auc = roc_auc
        save = True
    if np.mean(accuracy) > best_accuracy:
        best_accuracy = np.mean(accuracy)
    if epoch % 3 == 0:
        save = True
    print('save: ', save)

    # Print the report details
    print('Epoch {}:\n \
           train_log:\tLOSS: {:.3f}\ttri_loss: {:.3f}\tatt_loss: {:.3f}\thard_sample: {}\ttrain_time: {}\n \
           test_log:\tAUC: {:.3f}\tACC: {:.3f}+-{:.3f}\trecall: {:.3f}+-{:.3f}\tPrecision {:.3f}+-{:.3f}\t'.format(
        epoch + 1,
        avg_loss,
        avg_triplet_loss,
        avg_attention_loss,
        num_hard,
        (epoch_time_end - epoch_time_start) / 3600,
        roc_auc,
        np.mean(accuracy),
        np.std(accuracy),
        np.mean(recall),
        np.std(recall),
        np.mean(precision),
        np.std(precision),
    )
    )
    # Print the report for masked data
    print('Epoch {}:\n \
                   train_log:\tLOSS: {:.3f}\ttri_loss: {:.3f}\tatt_loss: {:.3f}\thard_sample: {}\ttrain_time: {}\n \
                   MASKED_LFW_test_log:\tAUC: {:.3f}\tACC: {:.3f}+-{:.3f}\trecall: {:.3f}+-{:.3f}\tPrecision {:.3f}+-{:.3f}\t'.format(
        epoch + 1,
        avg_loss,
        avg_triplet_loss,
        avg_attention_loss,
        num_hard,
        (epoch_time_end - epoch_time_start) / 3600,
        roc_auc_mask,
        np.mean(accuracy_mask),
        np.std(accuracy_mask),
        np.mean(recall_mask),
        np.std(recall_mask),
        np.mean(precision_mask),
        np.std(precision_mask),
    )
    )

    # Save the report as a file
    with open('logs/lfw_{}_log_tripletnotmaskV9.txt'.format(config['model']), 'a') as f:
        val_list = [
            'epoch: ' + str(epoch + 1) + '\t',
            'train:\t',
            'LOSS: ' + str('%.3f' % avg_loss) + '\t',
            'tri_loss: ' + str('%.3f' % avg_triplet_loss) + '\t',
            'att_loss: ' + str('%.3f' % avg_attention_loss) + '\t',
            'hard_sample: ' + str(num_hard) + '\t',
            'train_time: ' + str('%.3f' % ((epoch_time_end - epoch_time_start) / 3600))
        ]
        log = ''.join(str(value) for value in val_list)
        f.writelines(log + '\n')
        val_list = [
            'epoch: ' + str(epoch + 1) + '\t',
            'test:\t',
            'auc_masked: ' + str('%.3f' % roc_auc_mask) + '\t',
            'best_auc_MD: ' + str('%.3f' % best_roc_auc) + '\t',
            'acc_MD: ' + str('%.3f' % np.mean(accuracy_mask)) + '+-' + str('%.3f' % np.std(accuracy_mask)) + '\t',
            'best_acc_MD: ' + str('%.3f' % best_accuracy) + '\t',
            'recall_MD: ' + str('%.3f' % np.mean(recall_mask)) + '+-' + str('%.3f' % np.std(recall_mask)) + '\t',
            'precision_MD: ' + str('%.3f' % np.mean(precision_mask)) + '+-' + str(
                '%.3f' % np.std(precision_mask)) + '\t',
            'best_distances_MD: ' + str('%.3f' % np.mean(best_distances_mask)) + '+-' + str(
                '%.3f' % np.std(best_distances_mask)) + '\t',
            'tar_m: ' + str('%.3f' % np.mean(tar_mask)) + '\t',
        ]
        log = ''.join(str(value) for value in val_list)
        f.writelines(log + '\n')
        val_list = [
            'epoch: ' + str(epoch + 1) + '\t',
            'test:\t',
            'auc: ' + str('%.3f' % roc_auc) + '\t',
            'best_auc: ' + str('%.3f' % best_roc_auc) + '\t',
            'acc: ' + str('%.3f' % np.mean(accuracy)) + '+-' + str('%.3f' % np.std(accuracy)) + '\t',
            'best_acc: ' + str('%.3f' % best_accuracy) + '\t',
            'recall: ' + str('%.3f' % np.mean(recall)) + '+-' + str('%.3f' % np.std(recall)) + '\t',
            'precision: ' + str('%.3f' % np.mean(precision)) + '+-' + str('%.3f' % np.std(precision)) + '\t',
            'best_distances: ' + str('%.3f' % np.mean(best_distances)) + '+-' + str(
                '%.3f' % np.std(best_distances)) + '\t',
            'tar_m: ' + str('%.3f' % np.mean(tar)) + '\t',
        ]
        log = ''.join(str(value) for value in val_list)
        f.writelines(log + '\n')
        val_list = [
            'epoch: ' + str(epoch + 1) + '\t',
            'config:\t',
            'LR: ' + str(config['Learning_rate']) + '\t',
            'optimizer: ' + str(config['optimizer']) + '\t',
            'embedding_dim: ' + str(config['embedding_dim']) + '\t',
            'pretrained: ' + str(config['pretrained']) + '\t',
            'image_size: ' + str(config['image_size'])
        ]
        log = ''.join(str(value) for value in val_list)
        f.writelines(log + '\n' + '\n')

    # Save model weights
    if save:
        state = {
            'epoch': epoch + 1,
            'embedding_dimension': config['embedding_dim'],
            'batch_size_training': config['train_batch_size'],
            'model_state_dict': model.state_dict(),
            'model_architecture': config['model'],
            'optimizer_model_state_dict': optimizer_model.state_dict()
        }
        #
        if flag_train_multi_gpu:
            state['model_state_dict'] = model.module.state_dict()
        # For storing best euclidean distance threshold during LFW validation
        # if flag_validate_lfw:
        # state['best_distance_threshold'] = np.mean(best_distances)
        #
        torch.save(state, 'Model_training_checkpoints/model_{}_triplet_epoch_{}_rocNotMasked{:.3f}_rocMasked{:.3f}notmaskV9.pt'.format(config['model'],
                                                                                                     epoch + 1,
                                                                                                     roc_auc, roc_auc_mask))

# Training loop end
total_time_end = time.time()
total_time_elapsed = total_time_end - total_time_start
print("\nTraining finished: total time elapsed: {:.2f} hours.".format(total_time_elapsed / 3600))

