import torch
from tqdm import tqdm
import wandb
import torch.optim
from torch import nn
import numpy as np
from hyperparams import epochs,mini_batch_size,batch_size

class_loss = nn.CrossEntropyLoss()
def concord_cc2(r1, r2):
	mean_pred = torch.mean((r1 - torch.mean(r1))*(r2 - torch.mean(r2)))
	return 1 - (2*mean_pred)/(torch.var(r1) + torch.var(r2) + (torch.mean(r1)- torch.mean(r2))**2)
def train(train_loader, model, device, optimizer, epoch,scheduler = None):
    step = 0
    loop = tqdm(train_loader, leave=False)
    scaler = torch.cuda.amp.GradScaler()
    iters = len(train_loader)
    criterion = nn.MSELoss()
    for i,data in enumerate(loop):
        x = {}
        x['clip'] = data['clip'].to(device)
        valience_expected= data['valience'].to(device)
        arousal_expected = data['arousal'].to(device)
        # x['clip'] = torch.permute(x['clip'],(0,2,1,3,4))
        result = model(x['clip']).to(device)
        Class_V_L = class_loss(result[:,0:21],valience_expected).to(device).float()
        Class_A_L = class_loss(result[:,21:42],arousal_expected).to(device).float()
        loss= Class_V_L+Class_A_L
        wandb.log({
            "Train: Loss":loss.cpu().item(),
            "Train: arousal loss values":Class_A_L.cpu().item(),
            "Train: valence loss values": Class_V_L.cpu().item(),
            "Train: sample valence":result[:,0], 
            "Train: Sample Inputs":x['clip'],
            "Train: sample arousal":result[:,1],
            "epoch": epoch+1,
        })
        loss.sum().backward()
		# Update Optimizer
        optimizer.step()
        optimizer.zero_grad()
        loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
        loop.set_postfix(loss= loss.sum().item())
        step = step + 1

      
def val(val_loader, model, device, epoch):
    model.eval()
    loop = tqdm(val_loader, leave=False)
    loss = 0
    total = 0
    correct_v = 0
    correct_a = 0
    for data in loop:

        x = {}
        x['clip'] = data['clip'].to(device)
        wandb.log({
            "sample videos":
            wandb.Video(x['clip'][0].detach().cpu()*255,fps=2,format="webm")
        })
        # x['clip'] = torch.permute(x['clip'],(0,2,1,3,4))
        x['clip'] = x['clip'].permute((0,2,1,3,4))
        print(x['clip'].shape)
        valience_expected= data['valience'].to(device)
        arousal_expected = data['arousal'].to(device)
        
        with torch.no_grad():
            result = model(x['clip']).to(device)
            Class_V_L = class_loss(result[:,0:21].float(),valience_expected).to(device).float()
            Class_A_L = class_loss(result[:,21:42].float(),arousal_expected).to(device).float()
            loss+= (Class_V_L + Class_A_L).detach().item()
            valence_acutal = torch.argmax(result[:,0:21],1)

            arousal_actual = torch.argmax(result[:,21:42],1)
            correct_v = (valence_acutal== valience_expected).sum().item()
            correct_a = (arousal_actual== arousal_expected).sum().item()
            # loss+=CCC_V_L.detach().item()
            wandb.log({
            "Validation: epoch": epoch+1,
            "Validation: Valence Loss (minimize)": Class_V_L.item(),
            "Validation: Sample Inputs":x['clip'],
            "Validation: Arousal Loss (minimize)": Class_A_L.item()
            ,"Validation: loss (minimize)":(Class_A_L+Class_V_L).item(),
            "Validation: Sample Valence": result[:,0], 
            "Validation: Sample Arousal": result[:,1]
        })

        loop.set_description(f"Epoch [{epoch+1}/{epochs}] validation")
    wandb.log({"Accuracy (a)": (correct_a/(len(val_loader)*batch_size*.5)),"Accuracy (v)":(correct_v/(len(val_loader)*batch_size * .5))})
    model.train()
    return loss/(len(val_loader))


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
