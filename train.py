import sys
import time
import os
import torch
import torch.utils.data.dataset as dset

import folder
from fast_neural_style import network
from options import Options

import utils

opt = Options().parse()

if not os.path.exists(opt.log_dir):
    os.makedirs(opt.log_dir)
if opt.debug==False:
    sys.stdout = open(opt.log_dir + opt.log_file, "w")
args = vars(opt)
print('------------ Options -------------')
for k, v in sorted(args.items()):
    print('%s: %s' % (str(k), str(v)))
print('-------------- End ----------------')
print(opt)

model = network.PerceptualModel(opt)

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
print('#training images = %d' % dataset_size)



total_steps = 0
start_time = time.time()
for epoch in range(opt.num_epochs):
    epoch_iter = 0

    for i, data in enumerate(data_loader):
        epoch_iter += 1
        total_steps += 1
        generated_img = model.forward(data)
        content_score, style_score, tv_score = model.backward()

        if epoch_iter % opt.log_iter == 0:
            print("epoch_iter {}:".format(epoch_iter) +
                  'Content Loss: {:4f} Style Loss : {:4f} Total Variation Loss: {:4f}'.format(
                  #'Content Loss: {:4f} '.format(
                      content_score.data[0], style_score.data[0], tv_score.data[0]) +
                  #    content_score.data[0]) +
                  'Time Taken: %d sec' % (time.time() - start_time))
            start_time = time.time()
        model.generator_optimizer.step()

        # if total_steps % opt.display_freq == 0:
        #     visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt.save_iter == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            if not os.path.exists(opt.save_dir): 
                os.makedirs(opt.save_dir)
            utils.save(model.generator, opt.save_dir, 'generator', total_steps)

            # model.update_learning_rate()

            # TODO: vgg preprocess
            # TODO: weight init
            # TODO: visualize
if opt.debug==False:
    sys.stdout.close()
