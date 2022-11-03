from config.config_parameter import config2cfg
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
'''
check path and format
'''
class check_data():
    def __init__(self):
        self.cfg = config2cfg()
        self.train_data_path = r''
        self.val_data_path = r''
        self.file_path = r'' # the path is workspace path

    def check_path(self):
        data_root = os.path.join(self.cfg['workspace'], "mydata")
        self.train_data_path = os.path.join(data_root, "train")
        self.val_data_path = os.path.join(data_root, "val")
        # check
        if not os.path.exists(self.train_data_path):
            raise RuntimeError("train path is wrong!")
        if not os.path.exists(self.val_data_path):
            raise RuntimeError("val path is wrong!")
    def check_format(self):
        pass

'''
generated the txt label file 
'''
def make_txt_multiclass():
    num_label = 0
    cfg = config2cfg()
    path = cfg['dataroot'] # '/workspace/mydata/'

    for mode in os.listdir(path): # train or val file
        if os.path.exists(os.path.join(path, mode, 'label.txt')):
            os.remove(os.path.join(path, mode, 'label.txt'))  # delect the history file

    for mode in os.listdir(path): # train or val file
            # raise RuntimeError('delect the history label file!')
        for class_name in os.listdir(os.path.join(path, mode)):  # mask or nomask
            for line in os.listdir(os.path.join(path, mode, class_name)): # the image name of mask file or nomask file
                f = open(os.path.join(path, mode, "label.txt"), 'a')  # create the label file 'w':cover,'a':add
                f.write(path + '/' + mode + '/' + class_name + '/' + line + ' ' + str(num_label) + '\n')
                num_label += 1
                f.close()
    print('label file make success!')

'''
load myself data
'''
class load_mydata(Dataset):
    def __init__(self, img_path, transform=None):
        super(load_mydata, self).__init__()
        self.root = img_path
        self.txt_root = os.path.join(self.root, 'label.txt')
        f = open(self.txt_root, 'r')
        data = f.readlines()
        imgs = []
        labels = []
        for line in data:
            line = line.strip()
            word = line.split(' ')
            imgs.append(word[0])
            labels.append(word[1])
        self.img = imgs
        self.label = labels
        self.transform = transform

    def __len__(self):
        return len(self.label)

    def __getitem__(self, item):
        img = self.img[item]
        label = self.label[item]
        img = self.transform(Image.open(img).convert('RGB')) # write img and transform
        label = np.array(label).astype(np.int64)
        label = torch.from_numpy(label) # str-label -> tensor-label
        return img, label
'''
cfg = config2cfg()
data = load_mydata(img_path=os.path.join(cfg['dataroot'], 'train'), transform=transforms.Compose([transforms.Resize([224, 224]),
                                                                              transforms.ToTensor()])) # trainsforms.Compose([ , ])

data_loader = DataLoader(data, num_workers=0, batch_size=2, shuffle=True)
for num, img in enumerate(data_loader, 0):
    img = img[0][0]
    plt.imshow(img.permute(1, 2, 0))
    plt.show()
    # print(num)
'''



if __name__=='__main__':
    # mycheck = check_data()
    # mycheck.check_path()
    # make_txt_multiclass()
    pass

