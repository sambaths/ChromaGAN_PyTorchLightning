import config
import dataset
import utils

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import pytorch_lightning as pl
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')


torch.set_default_tensor_type('torch.FloatTensor')
bias=True

class discriminator_model(nn.Module):

  def __init__(self):
    super(discriminator_model, self).__init__()
    self.conv1 = nn.Conv2d(3, 64, kernel_size=(4,4),padding=1,stride=(2,2),bias=bias) # 64, 112, 112
    self.conv2 = nn.Conv2d(64, 128, kernel_size=(4,4), padding=1, stride=(2,2), bias=bias) # 128, 56, 56
    self.conv3 = nn.Conv2d(128,256, kernel_size=(4,4), padding=1, stride=(2,2), bias=bias) # 256, 28, 28, 2
    self.conv4 = nn.Conv2d(256,512, kernel_size=(4,4), padding=3, stride=(1,1), bias=bias) # 512, 28, 28
    self.conv5 = nn.Conv2d(512,1, kernel_size=(4,4), padding=3, stride=(1,1), bias=bias) # 1, 
    self.leaky_relu = nn.LeakyReLU(0.3)

  def forward(self,input):

    net = self.conv1(input)               #[-1, 64, 112, 112]
    net = self.leaky_relu(net)          #[-1, 64, 112, 112]    
    net = self.conv2(net)               #[-1, 128, 56, 56] 
    net = self.leaky_relu(net)          #[-1, 128, 56, 56] 
    net = self.conv3(net)               #[-1, 256, 28, 28]
    net = self.leaky_relu(net)          #[-1, 256, 28, 28]   
    net = self.conv4(net)               #[-1, 512, 27, 27]
    net = self.leaky_relu(net)          #[-1, 512, 27, 27]
    net = self.conv5(net)               #[-1, 1, 26, 26]
    return net

class colorization_model(nn.Module):
  def __init__(self):
    super(colorization_model, self).__init__()

    self.VGG_model = torchvision.models.vgg16(pretrained=True).float()
    self.VGG_model = nn.Sequential(*list(self.VGG_model.features.children())[:-8]) #[None, 512, 28, 28]
    self.relu = nn.ReLU()
    self.lrelu = nn.LeakyReLU(0.3)
    self.global_features_conv1 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=(2,2), bias=bias) #[None, 512, 14, 14]
    self.global_features_bn1 = nn.BatchNorm2d(512,eps=0.001,momentum=0.99)
    self.global_features_conv2 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias) #[None, 512, 14, 14]
    self.global_features_bn2 = nn.BatchNorm2d(512,eps=0.001,momentum=0.99)
    self.global_features_conv3 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=(2,2), bias=bias) #[None, 512, 7, 7]
    self.global_features_bn3 = nn.BatchNorm2d(512,eps=0.001,momentum=0.99)
    self.global_features_conv4 = nn.Conv2d(512, 512, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias) #[None, 512, 7, 7]
    self.global_features_bn4 = nn.BatchNorm2d(512,eps=0.001,momentum=0.99)

    self.global_features2_flatten = nn.Flatten()
    self.global_features2_dense1 = nn.Linear(512*7*7,1024)
    self.midlevel_conv1 = nn.Conv2d(512,512, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias) #[None, 512, 28, 28]
    self.global_features2_dense2 = nn.Linear(1024,512)
    self.midlevel_bn1 = nn.BatchNorm2d(512, eps=0.001,momentum=0.99)
    self.global_features2_dense3 = nn.Linear(512,256)
    self.midlevel_conv2 = nn.Conv2d(512,256, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)

    self.midlevel_bn2 = nn.BatchNorm2d(256,eps=0.001,momentum=0.99)
   
     #[None, 256, 28, 28]
    # self.midlevel_bn2 = nn.BatchNorm2d(256)#,,eps=0.001,momentum=0.99)

    self.global_featuresClass_flatten = nn.Flatten()
    self.global_featuresClass_dense1 = nn.Linear(512*7*7, 4096)
    self.global_featuresClass_dense2 = nn.Linear(4096, 4096)
    self.global_featuresClass_dense3 = nn.Linear(4096, 1000)
    self.softmax = nn.Softmax()

    self.outputmodel_conv1 = nn.Conv2d(512, 256, kernel_size=(1,1), padding=0, stride=(1,1),  bias=bias) 
    self.outputmodel_conv2 = nn.Conv2d(256, 128, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
    self.outputmodel_conv3 = nn.Conv2d(128, 64, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
    self.outputmodel_conv4 = nn.Conv2d(64, 64, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
    self.outputmodel_conv5 = nn.Conv2d(64, 32, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
    self.outputmodel_conv6 = nn.Conv2d(32, 2, kernel_size=(3,3), padding=1, stride=(1,1), bias=bias)
    self.outputmodel_upsample = nn.Upsample(scale_factor=(2,2))
    self.outputmodel_bn1 = nn.BatchNorm2d(128)
    self.outputmodel_bn2 = nn.BatchNorm2d(64)
    self.sigmoid = nn.Sigmoid()
    self.tanh = nn.Tanh()

  def forward(self,input_img):

    # VGG Without Top Layers

    vgg_out = self.VGG_model(input_img.float())

    #Global Features
    
    global_features = self.relu(self.global_features_conv1(vgg_out))  #[None, 512, 14, 14]
    global_features = self.global_features_bn1(global_features) #[None, 512, 14, 14]
    global_features = self.relu(self.global_features_conv2(global_features)) #[None, 512, 14, 14]
    global_features = self.global_features_bn2(global_features) #[None, 512, 14, 14]

    global_features = self.relu(self.global_features_conv3(global_features)) #[None, 512, 7, 7]
    global_features = self.global_features_bn3(global_features)  #[None, 512, 7, 7]
    global_features = self.relu(self.global_features_conv4(global_features)) #[None, 512, 7, 7]
    global_features = self.global_features_bn4(global_features) #[None, 512, 7, 7]
    
    global_features2 = self.global_features2_flatten(global_features) #[None, 512*7*7]
    
    global_features2 = self.global_features2_dense1(global_features2) #[None, 1024]
    global_features2 = self.global_features2_dense2(global_features2) #[None, 512]
    global_features2 = self.global_features2_dense3(global_features2) #[None, 256]
    global_features2 = global_features2.unsqueeze(2).expand(-1,256,28*28) #[None, 256, 784]
    global_features2 = global_features2.view((-1,256,28,28)) #[None, 256, 28, 28]

    global_featureClass = self.global_featuresClass_flatten(global_features) #[None, 512*7*7]
    global_featureClass = self.global_featuresClass_dense1(global_featureClass) #[None, 4096]
    global_featureClass = self.global_featuresClass_dense2(global_featureClass) #[None, 4096]
    global_featureClass = self.softmax(self.global_featuresClass_dense3(global_featureClass))#[None, 1000]
    
    # Mid Level Features
    midlevel_features = self.midlevel_conv1(vgg_out) #[None, 512, 28, 28]
    midlevel_features = self.midlevel_bn1(midlevel_features) #[None, 512, 28, 28]
    midlevel_features = self.midlevel_conv2(midlevel_features) #[None, 256, 28, 28]
    midlevel_features = self.midlevel_bn2(midlevel_features) #[None, 256, 28, 28]

    # Fusion of (VGG16 + MidLevel) + (VGG16 + Global)

    modelFusion = torch.cat([midlevel_features, global_features2],dim=1)

    # Fusion Colorization

    outputmodel = self.relu(self.outputmodel_conv1(modelFusion)) # None, 256, 28, 28
    outputmodel = self.relu(self.outputmodel_conv2(outputmodel)) # None, 128, 28, 28

    outputmodel = self.outputmodel_upsample(outputmodel) # None, 128, 56, 56
    outputmodel = self.outputmodel_bn1(outputmodel) # None, 128, 56, 56
    outputmodel = self.relu(self.outputmodel_conv3(outputmodel)) # None, 64, 56, 56
    outputmodel = self.relu(self.outputmodel_conv4(outputmodel)) # None, 64, 56, 56 

    outputmodel = self.outputmodel_upsample(outputmodel) # None, 64, 112, 112
    outputmodel = self.outputmodel_bn2(outputmodel) # None, 64, 112, 112
    outputmodel = self.relu(self.outputmodel_conv5(outputmodel)) # None, 32, 112, 112
    outputmodel = self.sigmoid(self.outputmodel_conv6(outputmodel)) # None, 2, 112, 112
    outputmodel = self.outputmodel_upsample(outputmodel) # None, 2, 224, 224

    return outputmodel, global_featureClass


class GAN(pl.LightningModule):
  def __init__(self, hparams):
    super(GAN, self).__init__()
    self.hparams = hparams

    self.netG = colorization_model()
    self.netD = discriminator_model()
    self.VGG_MODEL = torchvision.models.vgg16(pretrained=True).float()

    self.generated_imgs = None
    self.last_imgs = None

  def forward(self, input):

    self.predAB, self.classVector = self.netG(input)
    return self.predAB, self.classVector


  def wgan_loss(self, prediction, real_or_not):
    if real_or_not:
      return -torch.mean(prediction)
    else:
      return torch.mean(prediction)
  
  def gp_loss(self, y_pred, averaged_samples, gradient_penalty_weight):

    gradients = torch.autograd.grad(y_pred,averaged_samples,
                              grad_outputs=torch.ones(y_pred.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = (((gradients).norm(2, dim=1) - 1) ** 2).mean() * gradient_penalty_weight
    return gradient_penalty

  def training_step(self, batch, batch_idx, optimizer_idx):
    trainL, trainAB, _ = batch

    trainL_3 = torch.tensor(np.tile(trainL, [1,3,1,1])).float()
    trainL = torch.tensor(trainL).float()
    trainAB = torch.tensor(trainAB).float()

    predictVGG = F.softmax(self.VGG_MODEL(trainL_3))
    
    if optimizer_idx == 0:

      predAB, classVector = self(trainL_3)
      predLAB = torch.cat([trainL, predAB], dim=1)
      discpred = self.netD(predLAB)

      Loss_KLD = nn.KLDivLoss(size_average='False')(classVector.log(), predictVGG.detach()) * 0.003
      Loss_MSE = nn.MSELoss()(predAB, trainAB)
      Loss_WL = self.wgan_loss(discpred, True) * 0.1 
      Loss_G = Loss_KLD + Loss_MSE + Loss_WL

      tqdm_dict = {'Loss_G': Loss_G}
      output = OrderedDict({
          'loss': Loss_G,
          'progress_bar': tqdm_dict,
          'log': tqdm_dict
      })
      
      return output

    if optimizer_idx == 1:
      predLAB = torch.cat([trainL, self.predAB], dim=1)
      discpred = self.netD(predLAB.detach())
      realLAB = torch.cat([trainL, trainAB], dim=1)
      discreal = self.netD(realLAB)

      weights = torch.randn((trainAB.size(0),1,1,1))      
      if self.on_gpu:
        weights = weights.cuda(trainAB.device.index)    
      averaged_samples = (weights * trainAB ) + ((1 - weights) * self.predAB.detach())
      averaged_samples = torch.autograd.Variable(averaged_samples, requires_grad=True)
      avg_img = torch.cat([trainL, averaged_samples], dim=1)
      discavg = self.netD(avg_img)

      Loss_D_Fake = self.wgan_loss(discpred, False)
      Loss_D_Real = self.wgan_loss(discreal, True)
      Loss_D_avg = self.gp_loss(discavg, averaged_samples, config.GRADIENT_PENALTY_WEIGHT)
      Loss_D = Loss_D_Fake + Loss_D_Real + Loss_D_avg

      tqdm_dict = {'Loss_D': Loss_D}
      output = OrderedDict({
          'loss': Loss_D,
          'progress_bar': tqdm_dict,
          'log': tqdm_dict
      })
      return output
  def configure_optimizers(self):
    lr = self.hparams.lr
    b1 = self.hparams.b1
    b2 = self.hparams.b2

    optG = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(b1, b2))
    optD = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(b1, b2))
    return [optG, optD], []

  def train_dataloader(self):
    
    self.train_data = dataset.DATA(config.TRAIN_DIR)
    return torch.utils.data.DataLoader(self.train_data, 
                      batch_size=self.hparams.batch_size)

  def on_epoch_end(self):
    i=1
    utils.plot_some(self.train_data, self.netG, i)
    i+=1

    # for param in self.netD.parameters():
    #   param.requires_grad= False
    
    # predAB, classVector = self.netG(trainL_3)
    # predLAB = torch.cat([trainL, predAB], dim=1)
    # discpred = self.netD(predLAB)

    # return predAB, classVector, discpred

if __name__=='__main__':
  from argparse import Namespace
  from pytorch_lightning.callbacks import ModelCheckpoint
  import config
  args = {
      'batch_size': 2,
      'lr': 0.0002,
      'b1': 0.5,
      'b2': 0.999
  }
  hparams = Namespace(**args)
  gan_model = GAN(hparams)

  # most basic trainer, uses good defaults (1 gpu)
  checkpoint_callback = ModelCheckpoint(filepath=config.CHECKPOINT_DIR, save_top_k=-1,verbose=True, monitor='Loss_G')
  trainer = pl.Trainer(logger=False,checkpoint_callback=checkpoint_callback,max_epochs=10,default_save_path=config.CHECKPOINT_DIR)    
  trainer.fit(gan_model)   

  

