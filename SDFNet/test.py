import os
import utils
from PIL import Image
from torchvision import transforms
import numpy as np
import config
from model import SDFNet

import torch
from torch.autograd import Variable
import torch.optim as optim

model = SDFNet()
model_selection_path = config.testing['model_selection_path']
model.load_state_dict(torch.load(model_selection_path, map_location='cpu'))
model = torch.nn.DataParallel(model).cuda()
model.eval()

optimizer = optim.Adam(model.parameters(), lr=1e-4)

input_image_path = os.path.join('test_imgs','apple_example.jpeg')
fname = input_image_path.split(os.sep)[-1].split('.')[0]
input_size = config.data_setting['input_size']
img_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

# image_data = np.array(image_data.numpy())

out_dir = config.training['out_dir']
eval_task_name = config.testing['eval_task_name']
eval_dir = os.path.join(out_dir, 'eval')
eval_task_dir = os.path.join(eval_dir, eval_task_name)
cat_path = os.path.join(eval_task_dir, fname)
obj_path = os.path.join(cat_path, '%s.obj' % fname)
sdf_path = os.path.join(cat_path, '%s.dist' % fname)

box_size = config.testing['box_size']

with torch.no_grad():
    optimizer.zero_grad()
    image_data = Image.open(input_image_path).convert('RGB')
    image_data = img_transform(image_data)
    image_data = torch.unsqueeze(image_data, 0)
    image_data = Variable(image_data).cuda()
    utils.generate_mesh_sdf(image_data, model, obj_path, sdf_path, box_size=box_size)