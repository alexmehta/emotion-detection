from cleaner import clean_dataset
import torch
from train_def_afew import val, train
from loss_functions import *
from torch.utils.data.dataloader import DataLoader
from afew_network import TwoStreamAuralVisualModel
import wandb
import torch.optim
from afewdataset import AFEWDataset
from hyperparams import batch_size, num_workers, epochs, learning_rate,length
import argparse

# type: ignore
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description="Train Script")
parser.add_argument("--save_location",help="Save Location of models",default="new_baseline_models")
parser.add_argument("--load",help="load a epoch",default=None)
args = parser.parse_args()
wandb.init(project="baseline train afew")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(device)
train_set =clean_dataset(AFEWDataset(path='AFEW-TRAIN'))
val_set = clean_dataset(AFEWDataset(path='AFEW-TEST'))
print(len(train_set))
# train_set =AFEWDataset(path='AFEW-SMALL') 
# val_set = AFEWDataset(path='AFEW-SMALL')


train_loader = DataLoader(
    dataset=train_set, num_workers=num_workers, batch_size=batch_size, shuffle=True,pin_memory=True,drop_last=True)
val_loader = DataLoader(
    dataset=val_set, num_workers=num_workers, batch_size=batch_size, shuffle=True,pin_memory=True,drop_last=True)
model = TwoStreamAuralVisualModel().to(device)
if(args.load is not None):
    model.load_state_dict(torch.load(args.load))
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
wandb.config = {
    "learning_rate": learning_rate,
    "epochs": epochs,
    "batch_size": batch_size,
    "num_workers": num_workers
}
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
