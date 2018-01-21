import os
import argparse
import torch
from options import Options
import folder
import torch.utils.data.dataset as dset
from torch.autograd import Variable
import cv2
import torchvision
import numpy as np
from PIL import Image

opt = Options().parse()
from fast_neural_style import network

args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')

if not os.path.isfile(opt.model_path):
    print("=> no checkpoint found at '{}'".format(opt.model_path))
    exit(-1)

model = network.PerceptualModel(opt)
print("=> loading checkpoint '{}'".format(opt.model_path))
model.generator.load_state_dict(
    torch.load(opt.model_path, map_location=lambda storage, loc: storage.cuda(opt.gpu_ids[0])))
print(type(model))
print(type(model.generator))

def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    (a,b,c,d) = np.shape(img)
    img = np.reshape(img, (b,c,d))
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

datasets = []
for path in opt.data_roots:
    dataset = folder.ImageFolder(path, model.preprocess)
    datasets.append(dataset)
total_dataset = dset.ConcatDataset(datasets)
data_loader = torch.utils.data.DataLoader(total_dataset,
                                          batch_size=opt.batch_size,
                                          shuffle=(opt.mode == 'train'),
                                          drop_last=True,
                                          num_workers=opt.num_threads)
dataset_size = len(data_loader)
print('#testing images = %d' % dataset_size)

for i, data in enumerate(data_loader):
    model.image_tensor.resize_(data.size()).copy_(data)
    model.content_img = Variable(model.image_tensor)
    generated_img = model.generator.forward(model.content_img)
    print(generated_img)
    # save_image(opt.visualize_dir + '/' + str(i) + ".jpg", generated_img.data.cpu())
    torchvision.utils.save_image(generated_img.data.cpu(), opt.visualize_dir + '/' + str(i) + ".jpg")
    torchvision.utils.save_image(model.content_img.data.cpu(), opt.visualize_dir + '/' + str(i) + "_ori.jpg")
    if i > 100:
        break
