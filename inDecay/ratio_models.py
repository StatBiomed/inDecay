import torch
from torch import nn
import torchmetrics
import pytorch_lightning as pl

class Ratio_Model(pl.LightningModule):
    def __init__(self, channel_1d=32, channel_2d=4, k=3, lr=3e-3, optim_class='adam'):
        """
        Ratio Module predicts the ins-del ratio and mh - nonmh ratio with two inputs:
        1. onehot encoded guide sequences (20*4)
        2. substitution matrix with digonal filter (50,50)

        Params
        ---------
        channel_1d : int, the middle channel of conv1d module
        channel_2d : int, the middle channel of conv2d module
        k : int, applied to all kernel args incuding Conv*d and MaxPool
        lr : float, learning rate
        optim_class : str, adam, RMSprop
        """
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.optim_class = optim_class
        self.pearson_ins = torchmetrics.PearsonCorrCoef()
        self.pearson_mh = torchmetrics.PearsonCorrCoef()
        self.loss_fn = nn.BCELoss()

        # conv1d input : N,C,L
        self.sequence_module = nn.Sequential(
            nn.Conv1d(4, channel_1d, kernel_size=k),
            nn.Mish(),
            nn.Dropout(),
            nn.Conv1d(channel_1d,channel_1d, kernel_size=k),
            nn.Mish(),
            nn.Dropout()
        )
        # conv2d input : N, C, H, W
        self.matrix_module = nn.Sequential(
            nn.Conv2d(1,channel_2d, kernel_size=k),
            nn.Mish(),
            nn.Dropout(),
            nn.Conv2d(channel_2d,channel_2d, kernel_size=k),
            nn.Mish(),
            nn.MaxPool2d(kernel_size=k),
            nn.Dropout()
        )

        merged_channels = 1412
        self.fc_out = nn.Sequential(
            nn.Linear(merged_channels ,32),
            nn.Tanh(),
            nn.Linear(32, 2),
            nn.Sigmoid()
            )
    
    def resize_matrix(self, filtered_map):
        """
        crop or pad the second input matrix to desire size
        """
        # compute padding
        dim1, dim2 = tuple(filtered_map.shape)
        right_pad = max(self.matrix_size - dim2, 0)
        upper_pad = max(self.matrix_size - dim1, 0)
        padding = ( 
            0, right_pad,   
            upper_pad,0,
            0,0
        )

        # review tensor to (C, H, W) and then pad
        filtered_map = torch.from_numpy(filtered_map).unsqueeze(0).float()
        resize_map = nn.functional.pad(filtered_map, padding)
        return resize_map

    def forward(self, guide_oh, filtered_map):
        """
        Input
        -----
        guide_oh : tensor, [bs, 20, 4]
        filtered_map : tensor , [bs, 1, 50, 50]
        
        Return 
        ------
        out : [bs, 2], ins ratio and mh ratio
        """
        # feature encoding
        h_1d = self.sequence_module(guide_oh)
        h_2d = self.matrix_module(filtered_map)

        # transform hidden
        h_1d = torch.flatten(h_1d, start_dim=1)  #512 
        h_2d = torch.flatten(h_2d, start_dim=1)  #1024 for 20
        

        merged = torch.cat([h_1d, h_2d], dim=1)
        out = self.fc_out(merged)

        return out
    
    def configure_optimizers(self):
        lr = self.lr
        if self.optim_class == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=20,
                                          max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09)
        elif self.optim_class == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        return optimizer
    
    def training_step(self, batch, batch_index):
        
        guide_oh, filtered_map, ins_ratio, mh_ratio  = batch
        
        # (bs,2) , (bs,2)
        out = self.forward(guide_oh, filtered_map)
    
        ins_pred = out[:,0]
        mh_pred = out[:,1]

        loss_ins =  self.loss_fn(ins_pred, ins_ratio)
        loss_mh = self.loss_fn(mh_pred, mh_ratio)
        loss = loss_ins + loss_mh

        # logging
        with torch.no_grad():
            self.log_dict({
                "r_ins": self.pearson_ins(ins_pred, ins_ratio), 
                "r_mh": self.pearson_mh(mh_pred, mh_ratio), 
                "loss_ins" : loss_ins,
                "loss_mh" : loss_mh,
                "total_loss":loss
            })
        return loss

    def validation_step(self, batch, batch_index):
        guide_oh, filtered_map, ins_ratio, mh_ratio  = batch
        
        # (bs,2) , (bs,2)
        out = self.forward(guide_oh, filtered_map)
    
        ins_pred = out[:,0]
        mh_pred = out[:,1]

        loss_ins =  self.loss_fn(ins_pred, ins_ratio)
        loss_mh = self.loss_fn(mh_pred, mh_ratio)
        loss = loss_ins + loss_mh

        # logging
        with torch.no_grad():
            self.log_dict({
                "val_r_ins": self.pearson_ins(ins_pred, ins_ratio), 
                "val_r_mh": self.pearson_mh(mh_pred, mh_ratio), 
                "val_loss_ins" : loss_ins,
                "val_loss_mh" : loss_mh,
                "val_loss":loss
            })
    
    def test_step(self, batch, batch_index):
        return self.training_step(batch, batch_index)


class Ratio_Model_size20(Ratio_Model):
    def __init__(self, channel_1d=32, channel_2d=4, k=3, lr=3e-3, optim_class='adam'):
        """
        Ratio Module predicts the ins-del ratio and mh - nonmh ratio with two inputs:
        1. onehot encoded guide sequences (20*4)
        2. substitution matrix with digonal filter (20,20)

        Params
        ---------
        channel_1d : int, the middle channel of conv1d module
        channel_2d : int, the middle channel of conv2d module
        k : int, applied to all kernel args incuding Conv*d and MaxPool
        lr : float, learning rate
        optim_class : str, adam, RMSprop
        """
        super().__init__(channel_1d, channel_2d, k, lr, optim_class)

        self.matrix_module = nn.Sequential(
            nn.Conv2d(1,channel_2d, kernel_size=k),
            nn.Mish(),
            nn.Dropout(),
            nn.Conv2d(channel_2d,channel_2d, kernel_size=k),
            nn.Mish(),
            # nn.MaxPool2d(kernel_size=k),
            nn.Dropout()
        )

        merged_channels = 1536
        self.fc_out = nn.Sequential(
            nn.Linear(merged_channels ,32),
            nn.Tanh(),
            nn.Linear(32, 2),
            nn.Sigmoid()
            )