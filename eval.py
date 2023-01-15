from tqdm import tqdm
from torch import nn
import torch
from train_def import val
from tsav import TemporalConv_3dresNetwork 
from loss_functions import CCCEval
from aff2dataset import Aff2CompDataset
from torch.utils.data.dataloader import DataLoader
import os
from cleaner import clean_dataset
from strings import unpack_tuple
import numpy as np
def concord_cc2(y_true, y_pred, eps=1e-8):

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    vx = y_true - mean_true
    vy = y_pred - mean_pred
    cor = np.sum(vx * vy) / (np.sqrt(np.sum(vx**2)) * np.sqrt(np.sum(vy**2)))

    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred

    denominator = sd_true**2 + sd_pred**2 + (mean_true - mean_pred) ** 2

    return numerator / denominator
def load(set,device):
    loop = tqdm(set, leave=False)
    v_e = torch.zeros(len(set),1)
    v_a = torch.zeros(len(set),1)
    a_e = torch.zeros(len(set),1)
    a_a = torch.zeros(len(set),1)
    i = 0
    for data in loop:
        with torch.no_grad():
            input = data['clip'].to(device)
            # expected
            valience_expected= data['valience'].to(device)
            arousal_expected = data['arousal'].to(device)
            # output
            output = model(input).to(device)
            v_e[i] = valience_expected
            v_a[i] = output[0][0]
            a_e[i] = arousal_expected
            a_a[i] = output[0][1]
            i+=1

def eval(set, device,model_path="Final_Model.pth"):
    model = TemporalConv_3dresNetwork().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    loop = tqdm(set, leave=False)
    v_e = torch.zeros(len(set),1)
    v_a = torch.zeros(len(set),1)
    a_e = torch.zeros(len(set),1)
    a_a = torch.zeros(len(set),1)
    i = 0
    for data in loop:
        with torch.no_grad():
            input = data['clip'].to(device)
            # expected
            valience_expected= data['valience'].to(device)
            arousal_expected = data['arousal'].to(device)
            # output
            output = model(input).to(device)
            v_e[i] = valience_expected
            v_a[i] = output[0][0]
            a_e[i] = arousal_expected
            a_a[i] = output[0][1]
            i+=1
    return concord_cc2(v_e,v_a),concord_cc2(a_e,a_a)

def eval_all(set, device,directory='models',dialation=6,save_dir='results',length = 0):
    if(os.path.exists(f"{save_dir}/results_{dialation}.csv")):
        os.remove(f"{save_dir}/results_{dialation}.csv")
    f = open(f"{save_dir}/results_{dialation}.csv", "x")
    f.close()
    f = open(f"{save_dir}/results_{dialation}.csv", "w")
    f.write("File, Accuracy")
    f.close()
    list = os.listdir(directory)
    for file in list:
        acc =eval(set, device, model_path=str(
        os.path.join(directory, file)))
        with open(f'{save_dir}/results_{dialation}_{length}.csv', 'a') as filewr:
             filewr.write("\n" + str(file).removesuffix("_epoch_model.pth")+ ", " +
                          unpack_tuple(acc))
            
    f.close()

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    for i in range(1,100,5):
        test_set = Aff2CompDataset(root_dir='aff2_processed', mtl_path='mtl_data',dataset_dir='test_set.txt',test_set=True,dialation=i)
        length = len(test_set)
        print("dilation: ",i,"length",length)
        # test_loader = DataLoader(dataset=test_set, pin_memory=True,num_workers=4, batch_size=1,shuffle=False)
        # eval_all(test_loader, device,dialation=i,directory='models', save_dir='test_results',length = length)
