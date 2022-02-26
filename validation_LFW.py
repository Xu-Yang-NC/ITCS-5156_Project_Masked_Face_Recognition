import torch
import os
import numpy as np
from config_mask import config
import os
from validate_on_LFW import evaluate_lfw
from torch.nn.modules.distance import PairwiseDistance
import sys
from Data_loader.Data_loader_facenet_mask import train_dataloader, test_dataloader, LFWestMask_dataloader
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
pwd = os.path.abspath('./')

from Models.CBAM_Face_attention_Resnet_notmaskV3 import resnet18_cbam, resnet50_cbam, resnet101_cbam, resnet34_cbam, resnet152_cbam

model_path = os.path.join(pwd, 'Model_training_checkpoints')

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(pwd, 'Model_training_checkpoints')
x = [int(i.split('_')[4]) for i in os.listdir(model_path) if version in i]
x.sort()
for i in os.listdir(model_path):
    if (len(x)!=0) and ('epoch_'+str(x[-1]) in i) and (version in i):
        model_pathi = os.path.join(model_path, i)
        break

model_pathi = os.path.join(model_path, 'model_34_triplet_epoch_19_rocNotMasked0.918_rocMasked0.831notmaskV9.pt')
print(model_path)
if os.path.exists(model_pathi) and (version in model_pathi):
    if torch.cuda.is_available():
        model_state = torch.load(model_pathi)
    else:
        model_state = torch.load(model_pathi, map_location='cpu')
    model.load_state_dict(model_state['model_state_dict'])
    start_epoch = model_state['epoch']
    print('loaded %s' % model_pathi)
else:
    print('Model doesn't exist！')
    sys.exit(0)

if torch.cuda.is_available():
    model.cuda()

l2_distance = PairwiseDistance(2).cuda()

print("Validating on TestDataset! ...")
model.eval() 
with torch.no_grad():  
    distances, labels = [], []
    # print(1111111111, type(test_dataloader))
    # print(test_dataloader[0])
    # progress_bar = enumerate(tqdm(test_dataloader))
    # for batch_index, (data_a, data_b, label) in progress_bar:
    for batch_index, (data_a, data_b, label) in enumerate(test_dataloader):
       
        data_a = data_a.cuda()
        data_b = data_b.cuda()
        label = label.cuda()
        output_a, output_b = model(data_a), model(data_b)
        output_a = torch.div(output_a, torch.norm(output_a))
        output_b = torch.div(output_b, torch.norm(output_b))
        distance = l2_distance.forward(output_a, output_b)
        
        labels.append(label.cpu().detach().numpy())
        distances.append(distance.cpu().detach().numpy())
    
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for distance in distances for subdist in distance])
    true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
    tar, far = evaluate_lfw(
        distances=distances,
        labels=labels,
        epoch='',
        tag='NOTMaskedLFW_aucnotmask_valid',
        version=version,
        pltshow=True
    )

print("Validating on LFWMASKTestDataset! ...")
with torch.no_grad():  
    distances, labels = [], []
    # progress_bar = enumerate(tqdm(LFWestMask_dataloader))
    # for batch_index, (data_a, data_b, label) in progress_bar:
    for batch_index, (data_a, data_b, label) in enumerate(LFWestMask_dataloader):
       
        data_a = data_a.cuda()
        data_b = data_b.cuda()
        label = label.cuda()
        output_a, output_b = model(data_a), model(data_b)
        output_a = torch.div(output_a, torch.norm(output_a))
        output_b = torch.div(output_b, torch.norm(output_b))
        distance = l2_distance.forward(output_a, output_b)
        
        labels.append(label.cpu().detach().numpy())
        distances.append(distance.cpu().detach().numpy())
    
    labels = np.array([sublabel for label in labels for sublabel in label])
    distances = np.array([subdist for distance in distances for subdist in distance])
    true_positive_rate_mask, false_positive_rate_mask, precision_mask, recall_mask, \
    accuracy_mask, roc_auc_mask, best_distances_mask, \
    tar_mask, far_mask = evaluate_lfw(
        distances=distances,
        labels=labels,
        epoch='',
        tag='MaskedLFW_aucmask_valid',
        version=version,
        pltshow=True
    )
# Print report
print('LFW non-masked result test_log:\tAUC: {:.3f}\tACC: {:.3f}+-{:.3f}\trecall: {:.3f}+-{:.3f}\tPrecision {:.3f}+-{:.3f}\t'.format(
    roc_auc,
    np.mean(accuracy),
    np.std(accuracy),
    np.mean(recall),
    np.std(recall),
    np.mean(precision),
    np.std(precision))+'\tbest_distance:{:.3f}\t'.format(np.mean(best_distances))
)
print('\nLFW mask result test_log:\tAUC: {:.3f}\tACC: {:.3f}+-{:.3f}\trecall: {:.3f}+-{:.3f}\tPrecision {:.3f}+-{:.3f}\t'.format(
    roc_auc_mask,
    np.mean(accuracy_mask),
    np.std(accuracy_mask),
    np.mean(recall_mask),
    np.std(recall_mask),
    np.mean(precision_mask),
    np.std(precision_mask))+'\tbest_distance:{:.3f}\t'.format(np.mean(best_distances_mask))
)