import torch
from tqdm import tqdm
import wandb
import torch.optim
from torch import nn
import numpy as np
from hyperparams import epochs,mini_batch_size

def CCCLoss(y_hat, y, scale_factor=1., num_classes=2):
    y_hat_fl = torch.reshape(y_hat, (-1, num_classes))
    y_fl = torch.reshape(y, (-1, num_classes))

    yhat_mean = torch.mean(y_hat_fl, dim=0, keepdim=True)
    y_mean = torch.mean(y_fl, dim=0, keepdim=True)

    sxy = torch.mean(torch.mul(y_fl - y_mean, y_hat_fl - yhat_mean), dim=0)
    rhoc = torch.div(2 * sxy,
                     torch.var(y_fl, dim=0) + torch.var(y_hat_fl, dim=0) + torch.square(y_mean - yhat_mean) + 1e-8)

    return 1 - torch.mean(rhoc)
expression_classification_fn = nn.CrossEntropyLoss()
def concord_cc2(r1, r2):
	mean_pred = torch.mean((r1 - torch.mean(r1))*(r2 - torch.mean(r2)))
	return 1 - (2*mean_pred)/(torch.var(r1) + torch.var(r2) + (torch.mean(r1)- torch.mean(r2))**2)
def val_concord_cc2(r1, r2):
	mean_pred = torch.mean((r1 - torch.mean(r1))*(r2 - torch.mean(r2)))
	return (2*mean_pred)/(torch.var(r1) + torch.var(r2) + (torch.mean(r1)- torch.mean(r2))**2)




def pcc_ccc_loss(labels_th, scores_th):

    std_l_v = torch.std(labels_th[:,0]); std_p_v = torch.std(scores_th[:,0])
    std_l_a = torch.std(labels_th[:,1]); std_p_a = torch.std(scores_th[:,1])
    mean_l_v = torch.mean(labels_th[:,0]); mean_p_v = torch.mean(scores_th[:,0])
    mean_l_a = torch.mean(labels_th[:,1]); mean_p_a = torch.mean(scores_th[:,1])
   
    PCC_v = torch.mean( (labels_th[:,0] - mean_l_v) * (scores_th[:,0] - mean_p_v) ) / (std_l_v * std_p_v)
    PCC_a = torch.mean( (labels_th[:,1] - mean_l_a) * (scores_th[:,1] - mean_p_a) ) / (std_l_a * std_p_a)
#    PCC_v = torch.mean( (labels_th[:,0] - mean_l_v).t() @ (scores_th[:,0] - mean_p_v)/(std_l_v * std_p_v) )
#    PCC_a = torch.mean( (labels_th[:,1] - mean_l_a).t() @ (scores_th[:,1] - mean_p_a)/(std_l_a * std_p_a) )
    CCC_v = (2.0 * std_l_v * std_p_v * PCC_v) / ( std_l_v.pow(2) + std_p_v.pow(2) + (mean_l_v-mean_p_v).pow(2) )
    CCC_a = (2.0 * std_l_a * std_p_a * PCC_a) / ( std_l_a.pow(2) + std_p_a.pow(2) + (mean_l_a-mean_p_a).pow(2) )
   
    PCC_loss = 1.0 - (PCC_v + PCC_a)/2
    CCC_loss = 1.0 - (CCC_v + CCC_a)/2
    return PCC_loss, CCC_loss, CCC_v, CCC_a
def train(train_loader, model, device, optimizer, epoch,scheduler = None):
    step = 0
    loop = tqdm(train_loader, leave=False)
    scaler = torch.cuda.amp.GradScaler()
    iters = len(train_loader)
    for i,data in enumerate(loop):
        x = {}
        x['clip'] = data['clip'].to(device)
        valience_expected= data['valience'].to(device)
        arousal_expected = data['arousal'].to(device)
        # x['clip'] = torch.permute(x['clip'],(0,2,1,3,4))
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            result = model(x['clip']).to(device)
            CCC_V_L = concord_cc2(result[:,0],valience_expected).to(device)
            CCC_A_L = concord_cc2(result[:,1],arousal_expected).to(device)
        wandb.log({
            "Train: valence loss values":CCC_V_L.cpu().item(),"Train: total": CCC_V_L+CCC_A_L,
            "Train: arousal loss values":CCC_A_L.cpu().item(),
            "Train: sample valence":result[0,0], 
            "Train: sample arousal":result[0,1],

            "epoch": epoch+1,
        })
        CCC= ((2*CCC_V_L)+CCC_A_L)/(mini_batch_size)
        scaler.scale(CCC).backward()
        if ((i + 1) % mini_batch_size== 0) or (i + 1 == len(train_loader)):
		# Update Optimizer
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss= CCC.sum().item())
        step = step + 1
        if(scheduler!= None):
            scheduler.step(epoch + i /iters)

      
def val(val_loader, model, device, epoch):
    model.eval()
    loop = tqdm(val_loader, leave=False)
    loss = 0
    for data in loop:
        x = {}
        x['clip'] = data['clip'].to(device)
        # x['clip'] = torch.permute(x['clip'],(0,2,1,3,4))
        valience_expected= data['valience'].to(device)
        arousal_expected = data['arousal'].to(device)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                result = model(x['clip']).to(device)
                CCC_V_L = concord_cc2(result[:,0],valience_expected).to(device)
                CCC_A_L  = concord_cc2(result[:,1],arousal_expected).to(device)
                loss+= (CCC_V_L + CCC_A_L).detach().item()
                # loss+=CCC_V_L.detach().item()
        loop.set_description(f"Epoch [{epoch+1}/{epochs}] validation")
        wandb.log({
                "Validation: epoch": epoch+1,
                "Validation: Valence Loss (minimize)": CCC_V_L.item(),
                "Validation: Arousal Loss (minimize)": CCC_A_L.item()
                ,"Validation: loss (minimize)":(CCC_A_L+CCC_V_L).item(),
                "Validation: Sample Valence": result[0,0], 
                "Validation: Sample Arousal": result[0,1]
            })
    model.train()
    return loss/len(val_loader)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False
