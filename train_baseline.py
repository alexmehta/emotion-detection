from cleaner import clean_dataset
import torch
from train_def import val, train,EarlyStopper
from loss_functions import *
from torch.utils.data.dataloader import DataLoader
import wandb
import torch.optim
from aff2dataset import Aff2CompDataset
from hyperparams import batch_size, num_workers, epochs, learning_rate,length,mini_batch_size
import argparse
from actual_better import TwoStreamAuralVisualModel

# type: ignore
torch.backends.cudnn.benchmark = True
config = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "mini_batch_size":mini_batch_size,
    "num_workers": num_workers
}
parser = argparse.ArgumentParser(description="Train Script")
parser.add_argument("--save_location",help="Save Location of models",default="affwild_baseline_models_better")
parser.add_argument("--load",help="load a epoch",default=None)
args = parser.parse_args()
wandb.init(project="baseline train",config=config)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)
train_set = clean_dataset(Aff2CompDataset(
    root_dir='aff2_processed', mtl_path='mtl_data', dataset_dir='train_set.txt',vid_length=length))
val_set = clean_dataset(Aff2CompDataset(root_dir='aff2_processed', mtl_path='mtl_data',  dataset_dir='val_set.txt',vid_length=length))
train_loader = DataLoader(
    dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True,pin_memory=True,drop_last=True)
val_loader = DataLoader(
    dataset=val_set, num_workers=num_workers, batch_size=batch_size, shuffle=True,pin_memory=True,drop_last=True)
model = TwoStreamAuralVisualModel().to(device)
if(args.load is not None):
    model.load_state_dict(torch.load(args.load))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model.train()
wandb.watch(model)
best = [0.0,1000.0]
starting = 0
for epoch in range(starting,epochs):
    loss  = val(val_loader, model, device, epoch)
    wandb.log({"val loss total":loss})
    if(loss<=best[1]):
        best[0] = float(epoch+1)
        best[1] = float(loss)
        wandb.log({"best":best[0]}) 
    optimizer.zero_grad()
    train(train_loader, model, device, optimizer, epoch)
    optimizer.step()
    torch.save(model.state_dict(), f'{args.save_location}/{epoch+1}.pth')
print("best model: ",best)
model.train()
wandb.watch(model)
best = [0.0,1000.0]
starting = 0
for epoch in range(starting,epochs):
    loss  = val(val_loader, model, device, epoch)
    wandb.log({"val loss total":loss})
    if(loss<=best[1]):
        best[0] = float(epoch+1)
        best[1] = float(loss)
        wandb.log({"best":best[0]}) 
    optimizer.zero_grad()
    train(train_loader, model, device, optimizer, epoch)
    optimizer.step()
    torch.save(model.state_dict(), f'{args.save_location}/{epoch+1}.pth')
print("best model: ",best)