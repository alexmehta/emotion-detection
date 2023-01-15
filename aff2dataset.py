from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from clip_transforms import *
import csv
from torchvision.transforms import Compose
from torch.utils.data import DataLoader
from torchvision.io import write_video
from tqdm import tqdm
import math
from cleaner import clean_dataset
import torchaudio
import functools
class Aff2CompDataset(Dataset):

    def add_video(self,info,extracted_frames_list,transform=True):
        target = str(info['vid_name'][0])
        #if('/'.join(info['vid_name']) in self.cache):
            #return self.take_mask(*self.cache['/'.join(info['vid_name'])])
        for file in extracted_frames_list:
            if(str(file) in target):
                image_list = os.listdir(os.path.join(self.root_dir,"extracted",file))
                image_list.sort()
                for i,image in enumerate((image_list)):
                    if((info['vid_name'][1]) in image and "mask" not in str(file)):
                        store = info, file, image_list,i, image, transform
                        #if('/'.join(info['vid_name']) not in self.cache):
                            #self.cache['/'.join(info['vid_name'])] = store
                        return self.take_mask(*store)
        return None
    def take_mask(self, info, folder, image_list, current_idx, image,transform):
        before = current_idx
        after = len(image_list) - current_idx - 1
        info['start_frame'] = before
        info['end_frame'] = after
        info['path'] = os.path.join(self.root_dir,"extracted",folder,image)
        clip= np.zeros((self.vid_len, 112, 112, 3), dtype=np.uint8)
        needed_before =self.video_length* self.dialation 
        t = 0
        for z in range(current_idx-(self.video_length-1)*self.dialation,current_idx+1,self.dialation):
            try:
                image_path = os.path.join(self.root_dir,"extracted",folder,image_list[z])
                mask_img = Image.open(image_path)
                clip[t] = np.asarray(mask_img)
                t+=1
            except:
                pass
        return self.clip_transform(clip)

    def __init__(self,root_dir='aff2_processed',mtl_path = 'mtl_data/',test_set = False,target_frame = 7,vid_length = 8,dataset_dir = 'train_set.txt',dialation=6,ln_loop = False,non_uniform=None):
        super(Aff2CompDataset,self).__init__()
        self.bad = 0 
        self.ln = ln_loop
        self.total = 0
        self.target_frame = target_frame
        self.vid_len = vid_length
        self.video_length = vid_length
        self.test_set = test_set
        self.cache = {}
        self.remove = set()
        #file lists
        self.root_dir = root_dir

        # audio params
        self.window_size = 20e-3
        self.window_stride = 10e-3
        self.sample_rate = 44100
        num_fft = 2 ** math.ceil(math.log2(self.window_size * self.sample_rate))
        window_fn = torch.hann_window

        self.sample_len_secs = 12
        self.sample_len_frames = self.sample_len_secs * self.sample_rate
        self.audio_shift_sec = 5
        self.audio_shift_samples = self.audio_shift_sec * self.sample_rate

        self.audio_transform = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_mels=64,
                                                                    n_fft=num_fft,
                                                                    win_length=int(self.window_size * self.sample_rate),
                                                                    hop_length=int(self.window_stride
                                                                                   * self.sample_rate),
                                                                    window_fn=window_fn)
        self.audio_spec_transform = Compose([AmpToDB(), Normalize(mean=[-14.8], std=[19.895])])
        self.clip_transform = Compose([RandomClipFlip(),NumpyToPyTensor(),Normalize(mean=[0.472,0.363,0.331],std=[0.2697*2, 0.2399*2, 0.2309*2])])
        if(test_set):
            self.clip_transform  =  Compose([NumpyToPyTensor(),Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])])
        self.videos =   []
        self.videos += [each for each in os.listdir(root_dir) if each.endswith(".mp4")]
        self.metadata = []
        self.metadata += [each for each in os.listdir(root_dir) if each.endswith(".json")]
        self.audio_dir = []
        self.audio_dir +=[each for each in os.listdir(root_dir) if each.endswith(".wav")]
        self.extracted_frames = []
        self.extracted_frames += [each for each in os.listdir(os.path.join(root_dir , "extracted"))]
        
        #video info
        self.clip_len = vid_length
        self.input_shape = (112,112)
        self.dialation = dialation
        self.label_frame = self.clip_len * self.dialation
        csv = os.path.join(mtl_path, dataset_dir)
        self.dataset = []
        self.dataset += self.create_inputs(csv)
    def filter(self,arousal,valience, expression,):
        if(not self.test_set):
            return ((valience < 0) and expression==4) or (valience > 0 and expression==5) or (valience**2+ arousal**2  > 0.25 and expression==0)
        return False
    def create_inputs(self,csv_path):
        labels = []
        with open(csv_path) as csv_file:
            csv_reader = csv.reader(csv_file,delimiter=",")
            next(csv_reader)
            for row in csv_reader:
                labels.append(row)
        outputs = []
        for row in tqdm(labels):
            vid_name = row[0].split('/')
            valience = row[1]
            arousal = row[2]
            expressions = row[3]
            action_units = row[4:]
            expected_output = {}
            expected_output['vid_name'] = vid_name
            expected_output['valience'] = float(valience)
            expected_output['arousal'] = float(arousal)
            expected_output['expressions'] = int(expressions)
            expected_output['action_units'] = [int(action) for action in action_units]
            expected_output['frame_id'] = vid_name[1]
            if((expected_output['expressions']==-1 or expected_output['arousal']==-5.0 or expected_output['valience']==-5.0)):
                continue
            if((self.filter(expected_output['arousal'],expected_output['valience'],expected_output['expressions']))):
                continue
            if(not self.test_set and (int(expected_output['frame_id'].split('.')[0])<=10)):
                continue
            outputs.append(expected_output)
        return outputs
  
    def find_video(self,video_info):
        for video_name in self.videos:
            if(video_name.startswith(video_info[0])):
                return os.path.join(self.root_dir,video_name)

    def get_fps(self,video):
        video = cv2.VideoCapture(video)
        return video.get(cv2.CAP_PROP_FPS)
    def get_audio(self,info):
        try:
            audio_file = info['vid_name']
            # print(self.sample_rate * int(max((int(info['frame_id'].split('.')[0])-300)/30,0)))

            audio, sample_rate = torchaudio.load("aff2_processed/extracted/"+audio_file[0] + ".wav",num_frames=self.sample_len_frames,frame_offset=self.sample_rate * int(max((int(info['frame_id'].split('.')[0])-300)/30,0)))
            torchaudio.save('test.wav',audio,sample_rate)
            audio = self.audio_transform(audio).detach()

            return self.audio_spec_transform(audio)
        except:
            return None
    def __getitem__(self, index):
        d = self.dataset[index]
        dict = {}
        # dict['video_name'] = d['vid_name']
        dict['clip']  = self.add_video(d,self.extracted_frames)
        # dict['audio'] = self.get_audio(d)
        # dict['expressions'] = d['expressions']
        # dict['action_units'] = d['action_units']
        dict['valience'] = d['valience']
        dict['arousal'] = d['arousal']
        # dict['frame_id'] = d['frame_id']
        return dict
    def __len__(self):
        return len(self.dataset)
    def __add__(self,dict):
        self.dataset.append(dict) 
    def get_slice(self,audio,frame_offset,length):
        return audio[:,frame_offset:frame_offset+length]
  
    def __remove__(self,index):
        return self.dataset.pop(index)
if __name__ == "__main__":
    train_set = Aff2CompDataset(root_dir='aff2_processed')

    from hyperparams import length
    train_set = clean_dataset(Aff2CompDataset(
    root_dir='aff2_processed', mtl_path='mtl_data', dataset_dir='test_set.txt',vid_length=length))
     
    loader = DataLoader(dataset=train_set,batch_size=32,shuffle=True,pin_memory=True)
    channels_sum,channels_squares_sum, num_batches = 0,0,0
    to_sum = [0,2,3,4]
    for data in tqdm(loader):
        
        channels_sum += torch.mean(data['clip'],dim = to_sum)
        channels_squares_sum +=torch.mean(data['clip']**2,dim= to_sum)
        num_batches+=1
    mean = channels_sum/num_batches
    std = (channels_squares_sum/num_batches - mean**2)**0.5
    print(mean)
    print(std)
    
