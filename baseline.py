"""
Code from
"Two-Stream Aural-Visual Affect Analysis in the Wild"
Felix Kuhnke and Lars Rumberg and Joern Ostermann
Please see https://github.com/kuhnkeF/ABAW2020TNT
"""
import torch.nn as nn
import torch
from torchvision import models


class Dummy(nn.Module):
    def __init__(self):
        super(Dummy, self).__init__()

    def forward(self, input):
        return input


class VideoModel(nn.Module):
    def __init__(self, num_channels=3):
        super(VideoModel, self).__init__()
        self.r2plus1d = models.video.r2plus1d_18(pretrained=True)
        self.r2plus1d.fc = nn.Sequential(nn.Dropout(0.5),
                                         nn.Linear(in_features=self.r2plus1d.fc.in_features, out_features=17))
        self.modes = ["clip"]

    def forward(self, x):
        return self.r2plus1d(x)



class TwoStreamAuralVisualModel(nn.Module):
    def __init__(self, num_channels=3, audio_pretrained=False):
        super(TwoStreamAuralVisualModel, self).__init__()
        self.video_model = VideoModel(num_channels=num_channels)
        self.fc = nn.Sequential(nn.BatchNorm1d(self.video_model.r2plus1d.fc._modules['1'].in_features),nn.Dropout(0.3),nn.ReLU(),
                                          nn.Linear(in_features=self.video_model.r2plus1d.fc._modules['1'].in_features,
                                                    out_features=2))
        self.modes = ['clip' ]
        self.video_model.r2plus1d.fc = Dummy()
    def forward(self, x):
        video_model_features = self.video_model(x)
        out = self.fc(video_model_features)
        return out
def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        if(p.requires_grad==False):
            continue
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp       
if __name__ == '__main__':
    model = TwoStreamAuralVisualModel()
    print(get_n_params(model))
