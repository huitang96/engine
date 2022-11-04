# load the data import checkpoint test image
import torch
from PIL import Image
from torchvision import transforms

import torch.nn.parallel

model_save_path = r'/home/thui/projects/classification_proj/savefile/model.pth'
model = torch.load(model_save_path)

class_name = ['mask', 'nomask']
device = 'cuda:0' if torch.cuda.is_available() else 'cpu:0'

transforms_test = transforms.Compose([transforms.Resize([224, 224]),
                                      transforms.ToTensor()])
model.eval()
image_input = Image.open(r'/home/thui/projects/classification_proj/2.jpg')
image_tensor = transforms_test(image_input)
image_tensor.unsqueeze_(0)
image_tensor = image_tensor.to(device)
prediction = model(image_tensor)
for num in prediction:
    if num[0] > 0.5:
        pre = torch.tensor([0]).to(device)
        print(num[0].cpu().detach().numpy())
    if num[1] > 0.5:
        pre = torch.tensor([1]).to(device)
        print(num[1].cpu().detach().numpy())
print(class_name[pre])


