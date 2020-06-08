import model
import dataset
import utils
import config

import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

torch.set_default_tensor_type('torch.FloatTensor')
torch.set_default_dtype(torch.float)

class GAN(pl.LightningModule):
  def __init__(self, hparams):
    super(GAN, self).__init__()
    self.hparams = hparams

    self.netG = model.colorization_model()
    self.netD = model.discriminator_model()
    self.VGG_MODEL = torchvision.models.vgg16(pretrained=True)

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

    trainL = trainL.float()
    trainAB = trainAB.float()

    trainL_3 = torch.repeat_interleave(trainL, repeats=3, dim=1)
    
    if optimizer_idx == 0:

      predAB, classVector = self(trainL_3)
      predictVGG = nn.Softmax()(self.VGG_MODEL(trainL_3))
      predLAB = torch.cat([trainL, predAB], dim=1)
      for param in self.netD.parameters():
          param.requires_grad = False
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
      for param in self.netD.parameters():
          param.requires_grad = True
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
    
    utils.plot_some(self.train_data, self.netG, self.current_epoch)


if __name__=='__main__':

  from argparse import Namespace
  from pytorch_lightning.callbacks import ModelCheckpoint
  import config
  import torch
  args = {
      'batch_size': 10,
      'lr': 0.0002,
      'b1': 0.5,
      'b2': 0.999
  }
  hparams = Namespace(**args)
  gan_model = GAN(hparams)

  # most basic trainer, uses good defaults (1 gpu)
  checkpoint_callback = ModelCheckpoint(filepath=config.CHECKPOINT_DIR, save_top_k=-1,verbose=True)
  trainer = pl.Trainer(logging=False,progress_bar_refresh_rate=1,max_epochs=5, checkpoint_callback=checkpoint_callback)    
  trainer.fit(gan_model)   

