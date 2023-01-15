from tqdm import tqdm
def clean_dataset(set):
    i = 0
    removed = 0
    for z in tqdm(range(set.__len__()),leave=False):
        data = set.__getitem__(i)
        if(data['clip']==None):
            set.__remove__(i)
            i = i-1
            removed+=1
        i+=1
    print("removed: " ,removed)
    return set