# coding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
from torchvision import models, datasets, transforms
from torch.autograd import Variable
import numpy as np
import os
from PIL import Image
import matplotlib.image as mpimg
import pandas as pd
import matplotlib.pyplot as plt
from IoU import IoU
from replay_buffer import ReplayBuffer
import collections
from PIL import ImageDraw
import random
import re
import psutil
import gc
import train
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#----------------read data-----------------------
def Str2np(target):
    num = re.sub(r'\D', " ", target)
    a = np.fromstring(num, dtype=int, sep=' ')
    return a

def ReadData(csv_dir, img_dir, image_sets , test = None):
    imgs = []
    if test is not None:
        train_filename = csv_dir+'/val.csv'
    else:
        train_filename = csv_dir + '/train.csv'

    data = pd.read_csv(train_filename)
    name = data.fname.values
    child = data.child.values
    for i, entry in enumerate(name):
        pack = child[i]
        imgs.append([entry, Str2np(pack)])
    return imgs

root_dir = '/media/sjtu/831bebd9-c866-4ece-b878-5dbd68e5ca50/sjtu/data/VOC2012/'
csv_dir = os.path.join(root_dir,'src')
img_dir = os.path.join(root_dir,'JPEGImages')
image_sets =  ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',      'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',       'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
imgs = ReadData(csv_dir,img_dir,image_sets, test = None)
input_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean = [ 0.485, 0.456, 0.406 ],
                        std = [ 0.229, 0.224, 0.225 ]),
])

class Data(data.Dataset):
    def __init__(self, imgs, image_sets, input_transform = None,                  target_transform = None ,test = None):
        self.test = test
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.classes = len(image_sets)
        self.imgs = imgs

    def __getitem__(self, index):
        fn, pack = self.imgs[index]
        #label = torch.LongTensor([[y]])
        img = Image.open(fn).convert('RGB')
        #box = torch.LongTensor([[xmin, ymin, xmax, ymax]])
        if self.input_transform is not None:
            img = self.input_transform(img)
        return img, pack, fn

    def __len__(self):
        return len(self.imgs)

img_data = Data(imgs, image_sets,  input_transform = input_transform, target_transform=None                , test=None)
print(len(img_data.imgs))
img_batch = data.DataLoader(img_data, batch_size=1 ,shuffle=True, num_workers = 2)


# In[2]:

#--------load pre-trained model -------------
resnet = torch.load('resnet184classification.pth')
Extractor = nn.Sequential(*list(resnet.children())[:-2])

#--------define some functions-----------------
def Transition(box):
    x_img = img[:, :, box[1]:box[3]+1, box[0]:box[2]+1]
    return x_img

def roi_pooling(box_img):
    stride = [int(np.floor(box_img.size(2)/2.0)), int(np.floor(box_img.size(3)/2.0))]
    kernel_size = [int(np.ceil(box_img.size(2)/2.0)), int(np.ceil(box_img.size(3)/2.0))]
    padding = [0,0]
    if box_img.size(2) == 1:
        padding[0] = 1
        stride[0] = 1
        kernel_size[0] = 2
    if box_img.size(3) == 1:
        padding[1] = 1
        stride[1] = 1
        kernel_size[1] = 2
    padding = tuple(padding)
    stride = (stride[0], stride[1])
    kernel_size = (kernel_size[0], kernel_size[1])
    roi_pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride,                                 padding = padding, dilation=(1, 1))
    return roi_pooling(box_img)

def Cat_State(box_img, img):
    state1 = roi_pooling(box_img).view(-1)
    #state2 = roi_pooling(Extractor.forward(Variable(img, volatile = True))).view(-1)
    state2 = roi_pooling(Extractor.forward(Variable(img))).view(-1)
    state = torch.cat((state1, state2), 0).unsqueeze(0)
    return state

#-------------define hyper parameters------------
tau = 0.001
actor_lr = 0.01
critic_lr = 0.001
gamma = 0.99
buffer_size = 1000
minibatch_size = 20
max_episodes = 10
max_steps = 20
#concatenation of the feature vectors of global image(2048) and box image(2048)
state_dim = 4096
action_dim = 9
Penalty = 0.1
random_seed = 1234

print ' State Dimensions :  ', state_dim
print ' Action Dimensions :  ', action_dim
#--------------some function---------------#
A_STEP = 20
SCALE_STEP = 0.4
class Action(object):
    def __init__(self, box, img):
        self.x_step = img.size(3) / A_STEP
        self.y_step = img.size(2) / A_STEP
        self.enlarge = 1 + SCALE_STEP
        self.shrink = 1 - SCALE_STEP
        self.xmin, self.ymin, self.xmax, self.ymax = box
        self.center_x = (self.xmax + self.xmin) / 2
        self.center_y = (self.ymax + self.ymin) / 2
        self.width = self.xmax - self.xmin
        self.height = self.ymax - self.ymin
        self.narrow = self.width / 5
        self.box = box.clone()

    """def Check(self, xmin, ymin, xmax, ymax):
        if ymin <= 0:
            ymin = 0
        if ymax >= img.size(2):
            ymax = img.size(2)
        if xmin <= 0:
            xmin = 0
        if xmax >= img.size(3):
            xmax = img.size(3)
        return xmin, ymin, xmax, ymax"""

    def Up(self):
        ymin = self.ymin - self.y_step
        ymax = self.ymax - self.y_step
        #, ymin, _, ymax = self.Check(1, ymin, 1, ymax)
        new_box = torch.LongTensor([[self.xmin, ymin, self.xmax, ymax]])
        #print self.new_box
        return new_box, None

    def Down(self):
        ymin = self.ymin + self.y_step
        ymax = self.ymax + self.y_step
        #_, ymin, _, ymax = self.Check(1, ymin, 1, ymax)
        new_box = torch.LongTensor([[self.xmin, ymin, self.xmax, ymax]])
        return new_box, None

    def Left(self):
        xmin = self.xmin - self.x_step
        xmax = self.xmax - self.x_step
        #xmin, _, xmax, _ = self.Check(xmin, 1, xmax, 1)
        new_box = torch.LongTensor([[xmin, self.ymin, xmax, self.ymax]])
        return new_box, None

    def Right(self):
        xmin = self.xmin + self.x_step
        xmax = self.xmax + self.x_step
        #xmin, _, xmax, _ = self.Check(xmin, 1, xmax, 1)
        new_box = torch.LongTensor([[xmin, self.ymin, xmax, self.ymax]])
        return new_box, None

    def Enlarge(self):
        xmin = self.center_x - int(self.enlarge * self.width) / 2
        xmax = self.center_x + int(self.enlarge * self.width) / 2
        ymin = self.center_y - int(self.enlarge * self.height) / 2
        ymax = self.center_y + int(self.enlarge * self.height) /2
        #xmin, ymin, xmax, ymax = self.Check(xmin, ymin, xmax, ymax)
        new_box = torch.LongTensor([[xmin, ymin, xmax, ymax]])
        return new_box, None

    def Shrink(self):
        xmin = self.center_x - int(self.shrink * self.width) / 2
        xmax = self.center_x + int(self.shrink * self.width) / 2
        ymin = self.center_y - int(self.shrink * self.height) / 2
        ymax = self.center_y + int(self.shrink * self.height) / 2
        #xmin, ymin, xmax, ymax = self.Check(xmin, ymin, xmax, ymax)
        new_box = torch.LongTensor([[xmin, ymin, xmax, ymax]])
        return new_box, None

    def Narrow(self):
        xmin = self.xmin + self.narrow
        xmax = self.xmax - self.narrow
        #xmin, _, xmax, _ = self.Check(xmin, 1, xmax, 1)
        new_box = torch.LongTensor([[xmin, self.ymin, xmax, self.ymax]])
        return new_box, None

    def Stretch(self):
        xmin = self.xmin - self.narrow
        xmax = self.xmax + self.narrow
        #xmin, _, xmax, _ = self.Check(xmin, 1, xmax, 1)
        new_box = torch.LongTensor([[xmin, self.ymin, xmax, self.ymax]])
        return new_box, None

    def Trigger(self):
        # trigger flag is true, then initial
        return self.box, True

def Sample(box_, box, trigger, Trigger, Steps,img):
    terminal = None
    reward = 0.
    width = box[2] - box[0]
    height = box[3] - box[1]
    area = width * height
    if area < 20 or box[0] < action_bound[0] or     box[1] < action_bound[1] or box[2] > action_bound[2] or box[3] > action_bound[3]    or height < 15 or width < 15:
        box, trigger = Action(box,img).Trigger()
        reward -= Penalty
        box = box.squeeze(0)

    sign = np.zeros((num))
    iou = np.zeros((num))
    r = np.zeros((num))
    for k in xrange(num):
        iou[k] = IoU(box, ground_truth[k])
        r[k] = IoU(box, ground_truth[k]) - IoU(box_, ground_truth[k])
        sign[k] = float(np.sign(r[k]))
    reward += np.max(sign)

    if reward == 0.:
        reward -= Penalty
    if trigger is True:
        if np.max(iou) > 0.5:
            reward += 3.0
        else:
            reward -= 3.0
        Trigger += 1
        box, _ = Initial(img)
    else:
        Steps += 1

    box_img = Extractor(Variable(Transition(box)))
    next_state = Cat_State(box_img,img)

    if Trigger >= 4 or Steps >= max_steps:
        terminal = True
        Trigger = 0
        Steps = 0
    else:
        terminal = None
    return box, next_state, reward, terminal, Trigger, Steps

def Initial(img):
    num = random.randint(0, 3)
    box = torch.LongTensor([0,  0, img.size(3), img.size(2)])
    center_x = box[2] / 2
    center_y = box[3] / 2
    width = box[2]
    height = box[3]
    if num == 0:#center with the same scale
        box[0] = center_x - width / 4
        box[2] = center_x + width / 4
        box[1] = center_y - height / 4
        box[3] = center_y + height / 4
    elif num == 1: #center square with length equals to height
        box[0] = center_x - width / 4
        box[2] = center_x + width / 4
        box[1] = center_y - width / 4
        box[3] = center_y + width / 4
    elif num == 2: #center square with length equals to height
        box[0] = center_x - height / 4
        box[2] = center_x + height / 4
        box[1] = center_y - height / 4
        box[3] = center_y + height / 4
    else:      #rotate initial 1
        box[0] = center_x - height / 4
        box[2] = center_x + height / 4
        box[1] = center_y - width / 4
        box[3] = center_y + width / 4

    #box_img = Extractor.forward(Variable(Transition(box), volatile = True))
    box_img = Extractor.forward(Variable(Transition(box)))
    state = Cat_State(box_img,img)
    return box, state
#action_list = ['Up', 'Down', 'Left', 'Right', 'Enlarge' , 'Shrink', 'Narrow', 'Stretch', 'Trigger']
def Generate(a, box, img):
    if a == 0:
        return Action(box, img).Up()
    elif a == 1:
        return Action(box, img).Down()
    elif a == 2:
        return Action(box, img).Left()
    elif a == 3:
        return Action(box, img).Right()
    elif a == 4:
        return Action(box, img).Enlarge()
    elif a == 5:
        return Action(box, img).Shrink()
    elif a == 6:
        return Action(box, img).Narrow()
    elif a == 7:
        return Action(box, img).Stretch()
    else:
        return Action(box, img).Trigger()

def CreateDict(target, net):
    new_state_dict = collections.OrderedDict()
    #critic_target.load_state_dict( (1 - tau) * critic_target.state_dict() + tau * critic.state_dict())
    keys = net.state_dict().keys()
    for item in keys:
        params1 = net.state_dict()[item]
        params2 = target.state_dict()[item]
        new_state_dict[item] = (1 - tau) * params2 + tau * params1
    return new_state_dict

def CheckGradientUpdate(target, net):
    for param, shared_param in zip(net.parameters(),
                                                           target.parameters()):
        shared_param.data.copy_ ((1.0 - tau) * shared_param.data + tau * param.data)

def HardUpdate(target, net):
    for param, shared_param in zip(net.parameters(),
                                                       target.parameters()):
        shared_param.data.copy_(param.data)


# In[ ]:

ram = ReplayBuffer(buffer_size, random_seed)

trainer = train.Trainer(state_dim, action_dim, ram)


# In[ ]:

for i in xrange(max_episodes):
    img, pack, fn= next(iter(img_batch))
    num = pack.size(1) / 5
    ground_truth = torch.LongTensor(num, 4)
    cls = torch.LongTensor(num)
    pack = pack.squeeze(0)
    for j in xrange(num):
        ground_truth[j] = pack[5*j: 5*j+4].clone()
        cls[j] = pack[5*j+4]
    Trigger = 0
    Steps = 0
    box_, s = Initial(img)
    action_bound = torch.LongTensor([0, 0, img.size(3),img.size(2)])
    if num == 1:
        max_steps = 10
    else:
        max_steps = 20

    print 'EPISODE :  ', i
    while True:
        a_out = trainer.get_exploration_action(s)
        _, action = torch.max(a_out.data, 1)

        box, trigger = Generate(action.cpu().numpy()[0], box_, img)
        box, s2, r, t, Trigger, Steps = Sample(box_.squeeze(0), box.squeeze(0), trigger,                                                Trigger, Steps, img)
        print('{}\nreward: {}\n'.format(action, r))

        ram.add(s, a_out, r, t, s2)
        if trainer.ram.size() > minibatch_size:
            trainer.optimize()

        s = s2
        box_ = box.clone()

        if t is True:
            print ('terminated... ')
            break

    if i > 2:
        trainer.save_models(i)


print 'Completed episodes'


# print i
#
# img, pack, fn= next(iter(img_batch))
# num = int(pack.size(1) / 5)
# ground_truth = torch.LongTensor(num, 4)
# cls = torch.LongTensor(num)
# pack = pack.squeeze(0)
# for j in xrange(num):
#     ground_truth[j] = pack[5*j: 5*j+4].clone()
#     cls[j] = pack[5*j+4]
# Trigger = 0
# Steps = 0
# box_, s = Initial(img)
# action_bound = torch.LongTensor([0, 0, img.size(3),img.size(2)])
# print box_
#
#
# box, s2, r, t, Trigger, Steps = Sample(box_.squeeze(0), box.squeeze(0), trigger, \
#                                                Trigger, Steps, img)
# print Steps
#
# sign = np.zeros((num))
# iou = np.zeros((num))
# r = np.zeros((num))
# for k in xrange(num):
#     iou[k] = IoU(box, ground_truth[k])
#     r[k] = IoU(box, ground_truth[k]) - IoU(box_, ground_truth[k])
#     sign[k] = float(np.sign(r[k]))
# #reward += np.max(sign)
#
# print box, box_
#
#
# def PlotBox(img, box):
#     #img = Image.open(img)
#     draw = ImageDraw.Draw(img)
#     #plt.imshow(img_e)
#     draw.rectangle(box, outline = 'red')
#     #draw.rectangle((0,0, box[2], box[3]), outline = 'red')
#     plt.imshow(img)
#     plt.show()
#
# img_e = Image.open(fn[0])
#
# ground_truth = torch.LongTensor(num, 4)
# cls = torch.LongTensor(num)
# pack = pack.squeeze(0)
# for j in xrange(num):
#     ground_truth[j] = pack[5*j: 5*j+4].clone()
#     cls[j] = pack[5*j+4]
#     box1 = (ground_truth[j][0], ground_truth[j][1], ground_truth[j][2], ground_truth[j][3])
#     PlotBox(img_e ,box1)
#
# PlotBox(img_e, (box_[0], box_[1], box_[2], box_[3]))
# PlotBox(img_e, (box[0], box[1], box[2], box[3]))
#
#
#
# s1,a1,r1,t_batch, s2 = trainer.ram.sample_batch(1)
#
# a2 = trainer.target_actor.forward(s2.cuda()).detach()
# next_val = torch.squeeze(trainer.target_critic.forward(s2.cuda(), a2.cuda()).detach())
#
# s1,a1,r1,t_batch, s2 = trainer.ram.sample_batch(1)
# a2 = trainer.target_actor.forward(s2).detach()
# next_val = torch.squeeze(trainer.target_critic.forward(s2, a2).detach())
# y_expected = r1 + 0.99*next_val
# y_predicted = torch.squeeze(trainer.critic.forward(s1, a1))
#
# torch.save(trainer.actor, 'actor_'+str(i)+'.pth')
# torch.save(trainer.critic, 'critic_'+str(i)+'.pth')
# torch.save(trainer.target_actor, 'actor_target_'+str(i)+'.pth')
# torch.save(trainer.target_critic, 'critic_target_'+str(i)+'.pth')

# In[ ]:
