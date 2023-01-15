from tqdm import tqdm
from torch import nn
import torch
from train_def import val
from afew_network import TwoStreamAuralVisualModel
from loss_functions import CCCEval
from aff2dataset import Aff2CompDataset
from afewdataset import AFEWDataset
from torch.utils.data.dataloader import DataLoader
import os
from cleaner import clean_dataset
from strings import unpack_tuple
import wandb
batch_size = 6
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
                valence_acutal = torch.argmax(torch.nn.functional.softmax(output[:,0:21]),dim=1)
                arousal_actual = torch.argmax(torch.nn.functional.softmax(output[:,21:42]),dim=1)
                
                v_e[i*batch_size:(i*batch_size)+batch_size] = valience_expected
                v_a[i*batch_size:(i*batch_size)+batch_size] = valence_acutal
                a_e[i*batch_size:(i*batch_size)+batch_size]= arousal_expected
                a_a[i*batch_size:(i*batch_size)+batch_size] = arousal_actual
                # print(v_e[i*batch_size:(i*batch_size)+batch_size])
                # print(v_a[i*batch_size:(i*batch_size)+batch_size])
            else:
                valence_acutal = torch.argmax(output[:,0:21],1)
                arousal_actual = torch.argmax(output[:,21:42],1)
                v_e[i*batch_size:(i*batch_size)+output.size(0)] = valience_expected

                v_a[i*batch_size:(i*batch_size)+output.size(0)] =valence_acutal
                a_e[i*batch_size:(i*batch_size)+output.size(0)]= arousal_expected
                a_a[i*batch_size:(i*batch_size)+output.size(0)] = arousal_expected

    # v_e = torch.unsqueeze(v_e,1)
    # v_a = torch.unsqueeze(v_a,1)
    # a_e = torch.unsqueeze(a_e,1)
    # a_a = torch.unsqueeze(a_a,1)
    with open("test.txt",'a') as f:
        for i in range(v_e.size(0)):
            print(str(v_e[i].item())+ "," +str(v_a[i].item()), file=f)
    return concord_cc2(v_e,v_a),concord_cc2(a_e,a_a),((concord_cc2(v_e,v_a)+concord_cc2(a_e,a_a))/2)

def eval_all(set, device,directory='baseline_models',save_dir='results',dilation = 0):
    if(os.path.exists(f"{save_dir}/{dilation}_results.csv")):
        os.remove(f"{save_dir}/{dilation}_results.csv")
    f = open(f"{save_dir}/{dilation}_results.csv", "x")
    f.close()
    f = open(f"{save_dir}/{dilation}_results.csv", "w")
    f.write("File, Accuracy")
    f.close()
    list = os.listdir(directory)
    for file in list:
        print(os.path.join(directory,file))
        acc =eval(set, device, model_path=str(
        os.path.join(directory, file)))
        
        with open(f'{save_dir}/{dilation}_results.csv', 'a') as filewr:
             filewr.write("\n" + str(file).removesuffix("_epoch_model.pth")+ ", " +
                          unpack_tuple(acc))
            
    f.close()

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("USING CPU!!!")
    # wandb.init()
    for i in range(1,30):
        test_set = AFEWDataset(path='AFEW-TEST',dilation=i)
        # test_set = clean_dataset(Aff2CompDatasetNew(root_dir='aff2_processed', mtl_path='mtl_data',dataset_dir='test_set.txt',test_set=True,dialation=i))
        test_loader = DataLoader(dataset=test_set, pin_memory=True,num_workers=4, batch_size=batch_size,shuffle=False)
        eval_all(test_loader,device,dilation=i,save_dir="afew_tests_results") 
