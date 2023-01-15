from tqdm import tqdm
from torch import nn
import torch
from train_def import val
from actual_better import TwoStreamAuralVisualModel
from loss_functions import CCCEval
from aff2dataset import Aff2CompDataset
from torch.utils.data.dataloader import DataLoader
import os
from cleaner import clean_dataset
from hyperparams import length
from strings import unpack_tuple
import wandb
import argparse
batch_size = 16


def concord_cc2(r1, r2):
	mean_pred = torch.mean((r1 - torch.mean(r1))*(r2 - torch.mean(r2)))
	return (2*mean_pred)/(torch.var(r1) + torch.var(r2) + (torch.mean(r1)- torch.mean(r2))**2)


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
    model = TwoStreamAuralVisualModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    loop = tqdm(set, leave=False)
    v_e = torch.zeros(len(set)*batch_size)
    v_a = torch.zeros(len(set)*batch_size)
    a_e = torch.zeros(len(set)*batch_size)
    a_a = torch.zeros(len(set)*batch_size)
    i = 0
    for i,data in enumerate(loop):
        with torch.no_grad():
            input = data['clip'].to(device)
            # expected
            valience_expected= data['valience'].to(device).unsqueeze(0)
            arousal_expected = data['arousal'].to(device).unsqueeze(0)
            # output
            output = model(input).to(device)
            if(output.size(0)==batch_size):
                # print(range(i*batch_size,(i*batch_size)+batch_size))
                # print(valience_expected.shape)
                # print(v_e.shape)
                v_e[i*batch_size:(i*batch_size)+batch_size] = valience_expected
                v_a[i*batch_size:(i*batch_size)+batch_size] = output[:,0]
                a_e[i*batch_size:(i*batch_size)+batch_size]= arousal_expected
                a_a[i*batch_size:(i*batch_size)+batch_size] = output[:,1]
            else:
                # print(range(i*batch_size,(i*batch_size)+batch_size))
                # print(valience_expected.shape)
                # print(v_e.shape)
                v_e[i*batch_size:(i*batch_size)+output.size(0)] = valience_expected
                v_a[i*batch_size:(i*batch_size)+output.size(0)] = output[:,0]
                a_e[i*batch_size:(i*batch_size)+output.size(0)]= arousal_expected
                a_a[i*batch_size:(i*batch_size)+output.size(0)] = output[:,1]
    with open("test.txt",'a') as f:
        for i in range(v_e.size(0)):
            print(str(a_e[i].item()),str(a_a[i].item()),file=f)
    v_e = torch.unsqueeze(v_e,1)
    v_a = torch.unsqueeze(v_a,1)
    a_e = torch.unsqueeze(a_e,1)
    a_a = torch.unsqueeze(a_a,1)
    return concord_cc2(v_e,v_a),concord_cc2(a_e,a_a),((concord_cc2(v_e,v_a)+concord_cc2(a_e,a_a))/2)


def eval_all(set, device,directory='affwild_baseline_models_high_valence',save_dir='big_results',dilation = 0):
    if(os.path.exists(f"{save_dir}/{dilation}_results.csv")):
        os.remove(f"{save_dir}/{dilation}_results.csv")
    f = open(f"{save_dir}/{dilation}_results.csv", "x")
    f.close()
    f = open(f"{save_dir}/{dilation}_results.csv", "w")
    f.write("File, Accuracy")
    f.close()
    list = os.listdir(directory)
    for file in list:
        acc =eval(set, device, model_path=str(
        os.path.join(directory, file)))
        
        with open(f'{save_dir}/{dilation}_results.csv', 'a') as filewr:
             filewr.write("\n" + str(file).removesuffix(".pth")+ ", " +
                          unpack_tuple(acc))
            
    f.close()

if __name__ == '__main__':
    from hyperparams import length
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("USING CPU!!!")
    wandb.init()
    for i in range(6,7):
        # test_set = clean_dataset(AFEWDataset(dilation=i))
        test_set = clean_dataset(Aff2CompDataset(root_dir='aff2_processed', mtl_path='mtl_data',dataset_dir='test_set.txt',test_set=True,dialation=i,vid_length=length))
        test_loader = DataLoader(dataset=test_set, pin_memory=True,num_workers=2, batch_size=8,shuffle=True)
        eval_all(test_loader,device,dilation=i,directory='affwild_baseline_models_better') 