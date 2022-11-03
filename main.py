import torch
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from config.config_parameter import readconfig, config2cfg
from dataset.data import check_data, make_txt_multiclass, load_mydata
from network.alexnet import AlexNet
from torch.optim import optimizer
from matplotlib import pyplot as plt

def data_process():
    readcfg = readconfig()
    readcfg.read_config_file(r'/home/thui/projects/classification_proj')
    readcfg.get_main_proc_config('mode1')
    check = check_data()
    check.check_path()
    check.check_format()
    make_txt_multiclass()
    transform = transforms.Compose([transforms.Resize([224, 224]),
                                     transforms.ToTensor()])
    cfg = config2cfg()
    data_train = load_mydata(img_path=os.path.join(cfg['dataroot'], 'train'), transform=transform)
    data_loader_train = DataLoader(data_train, num_workers=0, batch_size=2, shuffle=True)
    data_val = load_mydata(img_path=os.path.join(cfg['dataroot'], 'val'), transform=transform)
    data_loader_val = DataLoader(data_val, num_workers=0, batch_size=2, shuffle=True)
    return data_loader_train, data_loader_val

if __name__=='__main__':
    data_loader_train, data_loader_val = data_process()
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'
    model = AlexNet(num_classes = 2).to(device)
    model.optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    model.loss_func = torch.nn.CrossEntropyLoss()
    model.train()
    epochs = 50
    loss_sum = 0.
    for epoch in range(1, epochs+1):
        for step, (feature, label) in enumerate(data_loader_train, 1):
            # feature, label = feature.to(device), label.to(device)
            model.optimizer.zero_grad()
            prediction = model(feature)
            loss = model.loss_func(prediction, label)
            loss.backward()
            model.optimizer.step() # 梯度更新

            loss_sum += loss
            if step == 5:
                print("epoch:", epoch, "loss:", loss_sum/step)












