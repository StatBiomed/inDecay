from turtle import forward
from typing import Any, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torchmetrics
import numpy as np
from qrguide import analysis_fn,transformation
from torch import Tensor, nn, softmax
from scipy.stats import pearsonr
import torch.nn.functional as F
from torch.distributions import multinomial
import pytorch_lightning as pl  # pytorch-lightning-1.7.7


class Topk_Event_Overlapping(torchmetrics.Metric):
    def __init__(self, k):
        """
        Metric to quantify the recall of the top 5 frequent events
        """
        super().__init__()
        self.k = k
        self.add_state("overlap", default=torch.tensor(0),
                       dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Union[list,torch.Tensor], target: Union[list,torch.Tensor]):
        # preds, target = self._input_format(preds, target)  # type: ignore
        # assert len(preds) == len(target)

        if isinstance(preds, torch.Tensor):
            preds = [preds.squeeze()]
            target = [target.squeeze()]

        for pred_i, target_i in zip(preds, target):

            assert pred_i.shape == target_i.shape
            k = self.k if self.k <= target_i.shape[0] else target_i.shape[0]

            pred_idxs = torch.topk(pred_i, k=k, dim=0).indices.cpu().numpy()
            target_idxs = torch.topk(target_i, k=k, dim=0).indices.cpu().numpy()

            batch_overlap = 0
            for i_p, i_t in zip(pred_idxs, target_idxs):
                batch_overlap += len(np.intersect1d(i_p, i_t))

            self.overlap += batch_overlap  # type: ignore
            self.total += 1  # type: ignore

    def compute(self):
        return self.overlap.float() / self.total  # type: ignore

class FrameShift_R2_557(torchmetrics.Metric):
    def __init__(self):
        """
        Metric to quantify the recall of the top 5 frequent events
        """
        super().__init__()

        self.add_state("pred", default=torch.Tensor([]), dist_reduce_fx="cat")
        self.add_state("target", default=torch.Tensor([]), dist_reduce_fx='cat')
        self.transform = torch.Tensor(transformation.get_fs_transform557('none'))
        self.Device_ = 'cpu'

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # preds, targets = self._input_format(preds, targets)  # type: ignore
        assert preds.shape == targets.shape
        if self.transform.shape[0] != targets.shape[1]:
            if targets.shape[1] == 536:
                self.transform  = self.transform[:-21]
            elif targets.shape[1] == 21:
                self.transform  = self.transform[-21:]
            else:
                raise ValueError("Invalid output shape")
        
        if self.Device_  != preds.device:
            self.Device_ = preds.device
            self.transform = self.transform.to(self.Device_)
            self.pred = self.pred.to(self.Device_)
            self.target = self.target.to(self.Device_)


        self.pred = torch.cat([self.pred, preds@self.transform], dim=0)
        self.target = torch.cat([self.target, targets@self.transform], dim=0)

    def compute(self):
        # if len(self.pred) < 2: 
        #     return torch.Tensor([0])
        # else:
        r = torch.corrcoef(torch.stack([self.pred, self.target]))[0,1]
        return r**2

class Base_ratio(pl.LightningModule):
    """
    The base ratio model that takes in one-hot encoded seq and then predict the ratio
    """

    def __init__(self, lr: Union[float, int] = 3e-4, L1_lambda: float = 0, L2_lambda: float = 1e-9, optim_class="Adam"):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.L1_lambda = L1_lambda
        self.L2_lambda = L2_lambda
        self.optim_class = optim_class

        self.train_r2 = torchmetrics.R2Score()
        self.val_r2 = torchmetrics.R2Score()
        self.test_r2 = torchmetrics.R2Score()
        self.loss_fn = nn.CrossEntropyLoss()

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

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
    
    def compute_L1(self):
        reg_loss = torch.sum(torch.abs(next(self.parameters())))
        return reg_loss

    def compute_L2(self):
        reg_loss = torch.sum(torch.pow(next(self.parameters()), 2))
        return reg_loss
    
    def training_step(self, train_batch, batch_index):

        X,Y = train_batch
        Ypred = self.forward(X)
        L1 = self.compute_L1()
        L2 = self.compute_L2()
        # cre = -1* torch.multiply(F.log_softmax(Ypred+1e-5, dim=1), Y).sum(dim=1).mean()
        cre = F.mse_loss(Y, Ypred)

        loss = cre + self.L1_lambda * L1 + self.L2_lambda * L2
        r2 = self.train_r2(Y[:,0], Ypred[:,0])
        
        self.log_dict({
            "train_r2":r2, "train_loss":loss
        })
        return loss
    
    def validation_step(self, val_batch, batch_index):

        X,Y = val_batch
        Ypred = self.forward(X)
        L1 = self.compute_L1()
        L2 = self.compute_L2()
        # cre = -1* torch.multiply(F.log_softmax(Ypred+1e-5, dim=1), Y).sum(dim=1).mean()
        cre = F.mse_loss(Y, Ypred)
        loss = cre + self.L1_lambda * L1 + self.L2_lambda * L2
        r2 = self.val_r2(Y[:,0], Ypred[:,0])
        
        self.log_dict({
            "val_r2":r2, "val_loss":loss
        })
        return loss
    
    def test_step(self, test_batch, batch_index):

        X,Y = test_batch
        Ypred = self.forward(X)
        L1 = self.compute_L1()
        L2 = self.compute_L2()
        # cre = -1* torch.multiply(F.log_softmax(Ypred+1e-5, dim=1), Y).sum(dim=1).mean()
        cre = F.mse_loss(Y, Ypred)
        loss = cre + self.L1_lambda * L1 + self.L2_lambda * L2
        r2 = self.test_r2(Y[:,0], Ypred[:,0])

        self.log_dict({
            "test_r2":r2, "test_loss":loss
        })
        return loss

class inDecay_ratio(Base_ratio):
    
    def __init__(self, lr: Union[float, int] = 3e-4, L1_lambda: float = 0, L2_lambda: float = 1e-9, optim_class="Adam"):
        """
        The linear version of ratio model
        takes in one-hot encoded seq and then predict the ratio
        """
        super().__init__(lr=lr, L1_lambda=L1_lambda, L2_lambda=L2_lambda, optim_class=optim_class)

        self.model = nn.Linear(60*4,2)

    def forward(self, X):
        X = torch.flatten(X, start_dim=1)
        return self.model(X)
    

class DeepDecay_ratio(Base_ratio):
    def __init__(self, out_channel, hidden, lr: Union[float, int] = 3e-4, L1_lambda: float = 0, L2_lambda: float = 1e-9, optim_class="Adam"):
        """
        The deep learning backbone ratio model takes in one-hot encoded seq and then predict the ratio

        Args
        -------
        out_channel : the number of CNN channels
        hidden : the number of hidden of MLP
        """
        super().__init__(lr=lr, L1_lambda=L1_lambda, L2_lambda=L2_lambda, optim_class=optim_class)

        self.model = nn.Sequential(
            nn.Conv1d(4, out_channel, kernel_size=8),
            nn.Mish(),
            nn.Flatten(start_dim=1),
            nn.Linear(53*out_channel, hidden),
            nn.Mish(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden,2)
        )

    def forward(self, X):
        X = torch.transpose(X, 1,2)
        return self.model(X)
    

class Base_del_model(pl.LightningModule):
    """
    The base pl model that takes in (x), y , c and then predict the probabity
    """

    def __init__(self, lr: Union[float, int] = 3e-4, L1_lambda: float = 0, L2_lambda: float = 1e-9, optim_class="Adam"):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.L1_lambda = L1_lambda
        self.L2_lambda = L2_lambda
        self.train_kld = torchmetrics.KLDivergence()
        self.val_kld = torchmetrics.KLDivergence()
        self.del_regressor = nn.Identity()
        self.top5_recall = Topk_Event_Overlapping(5)
        self.top10_recall = Topk_Event_Overlapping(10)
        self.optim_class = optim_class
    

    def configure_optimizers(self):
        lr = self.lr
        if self.optim_class == 'LBFGS':
            optimizer = torch.optim.LBFGS(self.parameters(), lr=lr, max_iter=20,
                                          max_eval=None, tolerance_grad=1e-07, tolerance_change=1e-09)
        else:
            optimizer = torch.optim.Adam(
                self.parameters(), lr=lr, weight_decay=self.L2_lambda)

        return optimizer

    def float_to_params(self, attr_name, attr, is_updatable):
        if attr is None:
            self.__setattr__(attr_name,
                             nn.parameter.Parameter(torch.rand(
                                 (1,)), requires_grad=is_updatable)
                             )
        else:
            init_parm = nn.parameter.Parameter(
                torch.Tensor([attr]).float(), requires_grad=is_updatable)
            self.__setattr__(attr_name, init_parm)

    def compute_Loss(self, out, y, reduce='mean'):
        """
        NLL loss 
        """
        # TODO: weight case by total count
        #  = 200 for > 200 or S
        if isinstance(y, list):
            cre_list = []
            for pred_i, y_i in zip(out, y):   #  iterate over each sample
                cre_list.append(
                    -1* torch.multiply(torch.log(pred_i+1e-5), y_i).sum()
                )
            
            cre = torch.stack(cre_list)

        elif isinstance(y, torch.Tensor):
            cre = -1* torch.multiply(torch.log(out+1e-5), y).sum()

        if reduce is None:
            cre = cre
        elif reduce == 'mean':
            cre = cre.mean()
        elif reduce == 'sum':
            cre = cre.sum()
        else:
            raise ValueError("invalid reduce of NLL loss")
        return cre

    def compute_L1(self):
        reg_loss = torch.sum(torch.abs(next(self.parameters())))
        return reg_loss

    def compute_L2(self):
        reg_loss = torch.sum(torch.pow(next(self.parameters()), 2))
        return reg_loss

    def mmej_forward(self, mmej_feat):
        raise NotImplementedError("please cover this")
        return None

    def nhej_forward(self, nhej_feat):
        raise NotImplementedError("please cover this")
        return None

    def normalize_y(self, y):

        with torch.no_grad():
            if isinstance(y, torch.Tensor):
                if torch.any(y.sum(1) != 1):
                    y = y / y.sum(dim=1, keepdim=True)

            elif isinstance(y, list):
                y_out = []
                for y_i in y:
                    if np.isclose(y_i.sum().item(), 1):
                        y_out.append(y_i)
                    else:
                        y_out.append(y_i / y_i.sum())
                y=y_out
        return y

    def forward(self, x):
        """
        Compute the probability for mmej and nhej separately and then combine them.
        During trainig, the ground turth mh ratio is used 
        """
        nhej_feat, mmej_feat, mh_strength = x
        mh_strength = mh_strength.view(-1, 1).float()
        mmej_p_pred = self.mmej_forward(mmej_feat)
        nhej_p_pred = self.nhej_forward(nhej_feat)
        y_pred = nhej_p_pred * (1-mh_strength) + mmej_p_pred * (mh_strength)
        # return torch.bmm(y_pred.unsqueeze(1), c_matrix).squeeze(), torch.bmm(y.unsqueeze(1), c_matrix).squeeze()
        return y_pred

    def predict(self, x, c_matrix=None):
        y_pred = self.del_regressor(x)
        if c_matrix is not None:
            y_pred = y_pred @ c_matrix
        return y_pred

    def training_step(self, train_batch, batch_idx):

        # forward
        X, y = train_batch
        if isinstance(y, list):
            p_pred = [self.forward(x) for x in X]
        else:
            p_pred = self.forward(X).unsqueeze(0)
        cre = self.compute_Loss(p_pred, y)

        # compute all kinds of loss and metrices
        y = self.normalize_y(y)

        L1 = self.compute_L1()
        L2 = self.compute_L2()

        
        if isinstance(y, list):
            mse = torch.stack([F.mse_loss(pred_i, y_i) for pred_i, y_i in zip(p_pred, y)]).mean()
            kld = torch.stack([self.train_kld(p.unsqueeze(0) + 1e-14, y_i.unsqueeze(0)+ 1e-14) for p, y_i in zip(p_pred, y)]).mean()
        else:
            mse = F.mse_loss(p_pred, y)
            kld = self.train_kld(p_pred+1e-14, y+1e-14)
        
        
        self.top5_recall(p_pred, y)
        self.top10_recall(p_pred, y)

        # logging
        self.log('train_mse', mse, sync_dist=False, batch_size=len(y))
        self.log('train_cre', cre, sync_dist=False, batch_size=len(y))
        self.log('train_L1', L1, sync_dist=False, batch_size=len(y))
        self.log('train_L2', L2, sync_dist=False, batch_size=len(y))
        self.log('train_kld', kld, batch_size=len(y))
        self.log('train_top5recall', self.top5_recall, batch_size=len(y))
        self.log('train_top10recall', self.top10_recall, batch_size=len(y))

        # the final loss is defined here
        loss = cre + L1*self.L1_lambda + L2*self.L2_lambda
        return loss

    def validation_step(self, train_batch, batch_idx):
        
        X, y = train_batch
        
        if isinstance(y, list):
            p_pred = [self.forward(x) for x in X]
        else:
            p_pred = self.forward(X).unsqueeze(0)

        cre = self.compute_Loss(p_pred, y)

        # compute all kinds of loss and metrices
        # if torch.any(y.sum(1) != 1):
        #     y = y / y.sum(dim=1, keepdim=True)
        y = self.normalize_y(y)

        if isinstance(y, list):
            mse = torch.stack([F.mse_loss(pred_i, y_i) for pred_i, y_i in zip(p_pred, y)]).mean()
            kld = torch.stack([self.train_kld(p.unsqueeze(0) + 1e-14, y_i.unsqueeze(0)+ 1e-14) for p, y_i in zip(p_pred, y)]).mean()
        else:
            mse = F.mse_loss(p_pred, y)
            kld = self.train_kld(p_pred+1e-14, y+1e-14)

        self.top5_recall(p_pred, y)
        self.top10_recall(p_pred, y)

        # logging
        self.log('val_mse', mse, batch_size=len(y))
        self.log('val_cre', cre, batch_size=len(y))
        self.log('val_kld', kld, batch_size=len(y))
        self.log('val_top5recall', self.top5_recall, batch_size=len(y))
        self.log('val_top10recall', self.top10_recall, batch_size=len(y))
        return cre

    def predict_step(self, batch, batch_idx):
        X, y = batch
        p_pred = [self.forward(x).unsqueeze(0) for x in X]
        return p_pred


class nhej_p_profiler(Base_del_model):
    """
    The CRSIPR deletion predictor. We model the deletion events as the combination of mmej and nhej events.

    Params
    -------------
    lr
            flaot, learning rate. Noted that LBFGS optimizer normally requires much larger learning rate
    kappa1
            float, exponential terms of mmej part
    mu & sigma
            float,
    kappa2 & c
            float, tranformation of precompute nhej_p. kappa2 :2, c : 2 `nhej_p =  \exp{10^c * p^kappa2} / Z` 
    updatable_param
            str

    Returns
    -------------
    model
    forward
    predict
    """

    def __init__(self,
                 lr: Union[float, int] = 1,
                 kappa1: Union[float, None] = None,
                 mu: Union[float, None] = None,
                 sigma: Union[float, None] = None,
                 kappa2: Union[float, None] = None,
                 c: Union[float, None] = None,
                 updatable_param="",
                 n_mmejfeat=3,
                 n_nhejfeat=6,
                 L1_lambda=0,
                 L2_lambda=1e-9,
                 optim_class='LBFGS'):
        super().__init__(lr=lr, L1_lambda=L1_lambda,
                         L2_lambda=L2_lambda, optim_class=optim_class)

        # new property
        self.updatable_param = updatable_param
        self.n_nhejfeat = n_nhejfeat
        self.n_mmejfeat = n_mmejfeat

        for attr_n, attr in zip(
                ['kappa1', 'mu', 'sigma', 'kappa2', 'c'], [
                    kappa1, mu, sigma, kappa2, c]
        ):
            is_updatable = attr_n in updatable_param
            self.float_to_params(attr_n, attr, is_updatable)

        # nhej deletion
        # mmej deletion
        self.mmej_regressor = nn.Sequential(
            nn.Linear(self.n_mmejfeat, 1),
            nn.Softmax(dim=1)
        )
        # self.mixture_act = nn.Sigmoid()

    def check_dimension(self, train_batch):
        """
        ?
        """
        if len(train_batch[2].shape) == 1:
            mm = [mmej_f.unsqueeze(0) for mmej_f in train_batch[0]]
            train_batch = [mm] + [ts.unsqueeze(0) for ts in train_batch[1:]]
        return train_batch

    def nhej_forward(self, nhej_p):
        """
        This model take in the precomputed p for nhej deletions, we can still apply some transformation here
        nhej_p: Tensor of shape [891,1], precomputed p for non-mh deletion events 
        """
        nhej_p_pred = 10**self.c * \
            torch.pow(nhej_p, self.kappa2)  # type: ignore
        return torch.softmax(nhej_p_pred, dim=1)

    def get_mask(self, mml):
        """
        only mmej events with max mh length >0 
        """
        with torch.no_grad():
            mask = mml > 0
        return mask

    def mmej_forward(self, mmej_feat_m):

        ss, mml, dl = torch.chunk(mmej_feat_m, 3, dim=-1)
        mask = self.get_mask(mml)

        # the first term
        ss_decay = 1/(1+torch.abs(ss))**self.kappa1
        # the third term
        dl_decay = ((dl-self.mu) / self.sigma)**2

        transformed_feat = torch.concat([ss_decay, mml, dl_decay], dim=-1)
        transformed_feat = torch.mul(transformed_feat, mask.long())
        mmej_p = self.mmej_regressor(transformed_feat).squeeze()
        return mmej_p


class nhej_feat_profiler(nhej_p_profiler):
    """
    The NHEJ part is predicted using extracted features.
    """

    def __init__(self, lr: Union[float, int] = 0.0003,
                 kappa1: Union[float, None] = None,
                 mu: Union[float, None] = None,
                 sigma: Union[float, None] = None,
                 kappa2: Union[float, None] = None,
                 kappa3: Union[float, None] = None,
                 c: Union[float, None] = None,
                 updatable_param="", n_mmejfeat=3, n_nhejfeat=6,
                 L1_lambda=0, L2_lambda=1e-9, optim_class='LBFGS'):
        super().__init__(lr, kappa1, mu, sigma, kappa2, c, updatable_param,
                         n_mmejfeat, n_nhejfeat, L1_lambda, L2_lambda, optim_class)

        self.float_to_params('kappa3', kappa3, 'kappa3' in updatable_param)
        self.nhej_regressor = nn.Sequential(
            nn.Linear(self.n_nhejfeat, 1),
            nn.Softmax(dim=1)
        )

    def nhej_forward(self, nhej_feat):
        """
        This model take in the precomputed p for nhej deletions, we can still apply some transformation here
        nhej_p: Tensor of shape [891,1], precomputed p for non-mh deletion events 
        """
        ss, dl, rb, lb, lc, rc = torch.chunk(
            nhej_feat, self.n_nhejfeat, dim=-1)
        # feature transform
        ss_decay = 1/(1+torch.abs(ss))**self.kappa2
        dl_decay = 1/(1+torch.abs(ss))**self.kappa3
        transformed_feat = torch.concat(
            [ss_decay, dl_decay, rb, lb, lc, rc], dim=-1)
        nhej_p = self.nhej_regressor(transformed_feat).squeeze()
        return nhej_p


class Linear_inDecay(Base_del_model):
    """
    Simply learning apply linear regression on the features. This model is used when features are transformed.
    Params
    -----------
    n_mmejfeat:
            int default 3, the input size for mmej mlp. Please specify when the number of features is changed due to transformation
    n_nhejfeat:
            int default 6, the input size for nhej mlp. Please specify when the number of features is changed due to transformation
    """

    def __init__(self,
                 hidden: list = [16],
                 lr=3e-4,
                 n_mmejfeat=3,
                 n_nhejfeat=6,
                 L1_lambda=0,
                 L2_lambda=1e-9,
                 optim_class='Adam'):
        super().__init__(lr=lr, L1_lambda=L1_lambda,
                         L2_lambda=L2_lambda, optim_class=optim_class)

        # add new class property
        self.n_nhejfeat = n_nhejfeat
        self.n_mmejfeat = n_mmejfeat

        # linear layer with softmax activation
        self.mmej_regressor = nn.Sequential(
            nn.Linear(self.n_mmejfeat, 1), nn.Softmax(dim=1))
        self.nhej_regressor = nn.Sequential(
            nn.Linear(self.n_nhejfeat, 1), nn.Softmax(dim=1))

    def get_mask(self, mml):
        """
        only mmej events with max mh length >0 
        """
        with torch.no_grad():
            mask = mml > 0
        return mask

    def nhej_forward(self, nhej_feat_m):
        Feat_TSs = torch.chunk(nhej_feat_m, self.n_nhejfeat, dim=-1)
        mask = self.get_mask(Feat_TSs[1]).squeeze()       # b, 891
        nhej_p = self.nhej_regressor(nhej_feat_m).squeeze()
        mask_p = torch.mul(nhej_p, mask.long())
        # return  mask_p / (mask_p.sum(dim=1, keepdim=True) + 1e-6) # to stablize the divid
        return torch.softmax(mask_p, dim=1)

    def mmej_forward(self, mmej_feat_m):
        ss, mml, dl = torch.chunk(mmej_feat_m, 3, dim=-1)
        mask = self.get_mask(mml).squeeze()       # b, 891
        # mmejpre = self.mmej_regressor(mmej_feat_m)

        mmej_p = self.mmej_regressor(mmej_feat_m).squeeze()  # b, 891
        mask_p = torch.mul(mmej_p, mask.long())
        # return mask_p / (mask_p.sum(dim=1, keepdim=True) + 1e-6)
        return torch.softmax(mask_p, dim=1)


class Old_Deep_inDecay(Linear_inDecay):
    """
    Deep learning version of inDecay for deletion events. Two mlps are used to model each mechanism. Prefer transformed features.
    Params
    ------------------
    hidden:
            list of int, the size and number of hidden layers
    n_mmejfeat:
            int default 3, the input size for mmej mlp. Please specify when the number of features is changed due to transformation
    n_nhejfeat:
            int default 6, the input size for nhej mlp. Please specify when the number of features is changed due to transformation
    """

    def __init__(self,
                 hidden: list = [16],
                 lr=3e-4,
                 n_mmejfeat=3,
                 n_nhejfeat=6,
                 L1_lambda=0,
                 L2_lambda=1e-9,
                 optim_class='Adam'):
        super().__init__(lr=lr, L1_lambda=L1_lambda,
                         L2_lambda=L2_lambda, optim_class=optim_class)

        dim_by_layer = [n_mmejfeat] + hidden + [1]
        dim_by_layer2 = [n_nhejfeat] + hidden + [1]

        # insert output layer with softmax activation
        mmej_mlp = [nn.Sequential(nn.Linear(din, dout, bias=True), nn.Mish())
                    for din, dout in zip(dim_by_layer[:-1], dim_by_layer[1:])]
        # mmej_mlp += [nn.Linear(hidden[-1], 1, bias=False), nn.Softmax(dim=1)]

        nhej_mlp = [nn.Sequential(nn.Linear(din, dout, bias=True), nn.Mish())
                    for din, dout in zip(dim_by_layer2[:-1], dim_by_layer2[1:])]
        # nhej_mlp += [nn.Linear(hidden[-1], 1, bias=False), nn.Softmax(dim=1)]

        self.mmej_regressor = nn.Sequential(*mmej_mlp)
        self.nhej_regressor = nn.Sequential(*nhej_mlp)

    def predict(self, x, c_matrix=None):
        y_pred = self.del_regressor(x)
        if c_matrix is not None:
            y_pred = y_pred @ c_matrix
        return y_pred


class ST_Decay(Base_del_model):
    """
    Linear inDecay model build on FORECasT coding system
    """

    def __init__(self, inputsize=9, outputsize=1, lr=3e-4, L1_lambda=3e-4, L2_lambda=3e-4):
        super().__init__(lr=lr, L1_lambda=L1_lambda, L2_lambda=L2_lambda)
        self.lr = lr
        self.del_regressor = nn.Linear(inputsize, outputsize)
        # self.loss_fn = nn.NLLLoss()
        self.l1_crit = nn.L1Loss(size_average=False)
        self.train_kld = torchmetrics.KLDivergence()
        self.val_kld = torchmetrics.KLDivergence()

    def forward(self, x):
        Out = self.del_regressor(x)  # [b, N_indel, 3633] -> [b, N_indel,1]
        y_pred = torch.softmax(Out.squeeze(2), dim=1)
        return y_pred


class ST_Decay_Scaler(Base_del_model):
    """
    Multi-layer DeepDecay model build on FORECasT coding system
    """

    def __init__(self, inputsize=9, outputsize=1, lr=3e-4, L1_lambda=3e-4, L2_lambda=3e-4):
        super().__init__(lr=lr, L1_lambda=L1_lambda, L2_lambda=L2_lambda)
        self.lr = lr
        self.del_regressor = nn.Linear(inputsize, outputsize)
        # self.loss_fn = nn.NLLLoss()
        self.l1_crit = nn.L1Loss(size_average=False)
        self.train_kld = torchmetrics.KLDivergence()
        self.val_kld = torchmetrics.KLDivergence()

    def get_scaler(self, x):
        with torch.no_grad():
            scaler_v = x[:, :, 7]
            del_ratio = scaler_v.max()
            ins_ratio = 1 - del_ratio
            scaler_v = torch.where(scaler_v > 0, del_ratio, ins_ratio)
        return scaler_v

    def forward(self, train_batch):
        x, y = train_batch

        Out = self.del_regressor(x)  # [b, N_indel, 3633] -> [b, N_indel,1]
        y_pred = torch.softmax(Out.squeeze(2), dim=1)

        scaler_v = self.get_scaler(x)
        scaled_y = torch.multiply(y_pred, scaler_v)

        return scaled_y / scaled_y.sum(), y


class Lindel_del(Base_del_model):
    """
    The module that uses Lindel X

    Since we have repeat the performance of Lindel prediction, we then validate whehh this mh-strength frame work will work
    """

    def __init__(self, inputsize, lr=3e-4,  L2_lambda: float = 0, L1_lambda: float = 0):
        super().__init__(lr=lr, L1_lambda=L1_lambda, L2_lambda=L2_lambda)

        self.del_regressor = nn.Sequential(
            nn.Linear(inputsize, 536),
            nn.Softmax(dim=1)
        )

    def forward(self, train_batch):

        (nhej_p, mmej_feat_m, mh_strength), x, y, c = train_batch

        # combined
        p = self.del_regressor(x)
        return p, y

class ST_DeepDecay(Base_del_model):
	"""
	inDecay's MLP model
	"""
	def __init__(self, inputsize=9, outputsize=1, hidden=[16], lr=3e-4, L1_lambda=3e-4, L2_lambda=3e-4):
		super().__init__(lr=lr, L1_lambda=L1_lambda, L2_lambda=L2_lambda)
		self.lr = lr
		layer_size = [inputsize] + hidden 

		layer_ls = [nn.Sequential(nn.Linear(din, dout, bias=True), nn.Mish())
				for din, dout in zip(layer_size[:-1] , layer_size[1:])]
		layer_ls += [nn.Linear(hidden[-1],1)]
		self.del_regressor = nn.Sequential(*layer_ls)


		self.l1_crit = nn.L1Loss(size_average=False) 
		self.train_kld = torchmetrics.KLDivergence()
		self.val_kld = torchmetrics.KLDivergence()
	
	def forward(self, x):
		Out = self.del_regressor(x) # [b, N_indel, 3633] -> [b, N_indel,1]
		y_pred = torch.softmax(Out.squeeze(), dim=0)
		return y_pred

class ST_delfeat_DeepDecay(ST_DeepDecay):
	"""
	DeepDecay multi-ev model
	"""
	def __init__(self, del_feat, inputsize=9, outputsize=1, hidden=[16], lr=3e-4, L1_lambda=3e-4, L2_lambda=3e-4):
		super().__init__(inputsize=inputsize, outputsize=outputsize, hidden=hidden, lr=lr, L1_lambda=L1_lambda, L2_lambda=L2_lambda)

        # deletion model
		layer_size = [del_feat] + hidden 

		layer_ls = [nn.Sequential(nn.Linear(din, dout, bias=True), nn.Mish())
				for din, dout in zip(layer_size[:-1] , layer_size[1:])]
		layer_ls += [nn.Linear(hidden[-1],1)]
          
		self.del_regressor = nn.Sequential(*layer_ls)


		# insertion model
		layer_ins = [inputsize-del_feat] + hidden

		layer_ls = [nn.Sequential(nn.Linear(din, dout, bias=True), nn.Mish())
				for din, dout in zip(layer_ins[:-1] , layer_ins[1:])]
		layer_ls += [nn.Linear(hidden[-1],1)]
          
		self.del_regressor = nn.Sequential(*layer_ls)
		self.del_feat = del_feat

	def forward(self, x):
		x_del = x[:,:,:self.del_feat]
		x_ins = x[:,:,:self.del_feat]
		Out = self.del_regressor(x) # [b, N_indel, 3633] -> [b, N_indel,1]
		y_pred = torch.softmax(Out.squeeze(2), dim=1)
		return y_pred


class ST_DeepDecay_dropout(ST_DeepDecay):
	"""
	repeat Lindel's linear model
	"""
	def __init__(self, inputsize=9, outputsize=1, hidden=[16], lr=3e-4, L1_lambda=3e-4, L2_lambda=3e-4):
		super().__init__(inputsize=inputsize, outputsize=outputsize, hidden=hidden, lr=lr,L1_lambda=L1_lambda,L2_lambda=L2_lambda)

		layer_size = [inputsize] + hidden 

		layer_ls = [nn.Sequential(nn.Linear(din, dout, bias=True), nn.Mish(), nn.Dropout(p=0.3))
				for din, dout in zip(layer_size[:-1] , layer_size[1:])]
		layer_ls += [nn.Linear(hidden[-1],1)]
		self.del_regressor = nn.Sequential(*layer_ls)


class ST_DeepDecay_Multinomial(ST_DeepDecay):
	"""
	repeat Lindel's linear model
	"""
	def __init__(self, inputsize=9, outputsize=1, hidden=[16], lr=3e-4, L1_lambda=3e-4, L2_lambda=3e-4):
		super().__init__(inputsize=inputsize, outputsize=outputsize, hidden=hidden, lr=lr,L1_lambda=L1_lambda,L2_lambda=L2_lambda)

	def compute_Loss(self, out, y):
		"""
		Multi-nomial loss 
		"""
		with torch.no_grad():
			total_count = int(y.sum().item())
			reg = np.sqrt(total_count)
		Multino = multinomial.Multinomial(total_count=total_count, probs=out)
		loss = -1 * Multino.log_prob(y)

		# trunc to 200 / 1000

		# down-sampling 
		return loss /reg

	def compute_cre(self, out, y_norm):
		"""compute NLL loss

		Args:
			out (_type_): predicted outcome
			y_norm (_type_): normlized y
		"""
		cre = super().compute_Loss(out, y_norm)
		return cre

	def training_step(self, train_batch, batch_idx):
		
		# forward
		p_pred, y = self.forward(train_batch)
		cre = self.compute_Loss(p_pred, y)

		# compute all kinds of loss and metrices
		y_norm = y / y.sum()
		L1 = self.compute_L1()
		L2 = self.compute_L2()
		mse = F.mse_loss(p_pred, y_norm) 
		
		kld = self.train_kld(p_pred+1e-14, y_norm+1e-14)
		self.top5_recall(p_pred, y)
		self.top10_recall(p_pred, y)

		# logging
		self.log('train_mse', mse, sync_dist=True)
		self.log('train_cre', cre, sync_dist=True)
		self.log('train_L1', L1, sync_dist=True)
		self.log('train_L2', L2, sync_dist=True)
		self.log('train_kld', self.train_kld)
		self.log('train_top5recall', self.top5_recall)
		self.log('train_top10recall', self.top10_recall)

		##  the final loss is defined here
		loss = cre + L1*self.L1_lambda + L2*self.L2_lambda
		return loss
	
	def validation_step(self, train_batch, batch_idx):

		p_pred, y = self.forward(train_batch)
		y_norm = y / y.sum()
		# compute all kinds of loss and metrices
		# cre = self.compute_Loss(p_pred, y)
		cre = self.compute_cre(p_pred, y_norm)
		mse = F.mse_loss(p_pred, y_norm) 
		kld = self.val_kld(p_pred+1e-14, y_norm+1e-14)
		self.top5_recall(p_pred, y)
		self.top10_recall(p_pred, y)

		# logging
		self.log('val_mse', mse)
		self.log('val_cre', cre)
		self.log('val_kld', self.val_kld)
		self.log('val_top5recall', self.top5_recall)
		self.log('val_top10recall', self.top10_recall)
		return cre
	
	def predict_step(self, batch, batch_idx):
		p_pred, y = self.forward(batch)
		return p_pred
	

class Lindel_X_mh(Base_del_model):
    """
    The module that uses Lindel X

    Since we have repeat the performance of Lindel prediction, we then validate whehh this mh-strength frame work will work
    """

    def __init__(self, inputsize, lr=3e-4,  L2_lambda: float = 0, L1_lambda: float = 0):
        super().__init__(lr=lr, L1_lambda=L1_lambda, L2_lambda=L2_lambda)
        self.kappa = nn.parameter.Parameter(
            torch.Tensor([2.3]), requires_grad=True)
        self.del_regressor = nn.Sequential(
            nn.Linear(inputsize, 536),
            nn.Softmax(dim=1)
        )

    def forward(self, train_batch):

        (nhej_p, mmej_feat_m, mh_strength), x, y, c = train_batch
        mh_strength = mh_strength.view(-1, 1)
        mmej_p = self.del_regressor(x)

        # nhej part
        nhej_p = nhej_p**self.kappa
        nhej_p = F.softmax(nhej_p, dim=1)

        # combined
        overal_p = nhej_p * (1-mh_strength) + mmej_p * (mh_strength)
        return overal_p, y

# old models


class deletion_decay_profiler(pl.LightningModule):
    """
    The CRSIPR deletion predictor. We model the deletion events as 
    """
    def __init__(self,
                    beta: float = None,   # type: ignore
                    kappa: float = None,  # type: ignore
                    mu: float = None,     # type: ignore
                    sigma: float = None,  # type: ignore
                    update_hyper: bool = False):
        super().__init__()
        self.update_hyper = update_hyper

        for attr_n, attr in zip(
            ['kappa', 'mu', 'sigma'], [kappa, mu, sigma]
        ):
            self.float_to_params(attr_n, attr)

        self.mmej_regressor = nn.Sequential(
                    nn.Linear(3, 1), 
                    nn.Softmax(dim=1)
                    )
        self.kld_fn = torchmetrics.KLDivergence()
    
    def float_to_params(self, attr_name, attr):
        if attr is None:
            self.__setattr__(attr_name,
                nn.parameter.Parameter(
                    torch.randn((1,)), requires_grad=True)
                    )
        else:
            self.__setattr__(attr_name,
                            nn.parameter.Parameter(
                                torch.Tensor([attr]), requires_grad=False)
                            )

    def mmej_forward(self, mmej_feat_m):

        ss, mml, dl = torch.chunk(mmej_feat_m, 3, dim=-1)

        # the first term
        ss_decay = torch.exp(1/(1+ss))*self.kappa
        # the third term
        # dl_decay = torch.exp(
        #    -1* (dl-self.mu)**2 / torch.pow(self.sigma, 2)
        # )
        dl_decay =  (dl-self.mu)**2 / torch.pow(self.sigma, 2)

        transformed_feat = torch.concat([ss_decay, mml, dl_decay], dim=-1)
        mmej_p = self.mmej_regressor(transformed_feat)
        return mmej_p
    
    def forward(self, x):
        nhej_p, mmej_feat_m, mh_strength = x
        mmej_p = self.mmej_forward(mmej_feat_m).squeeze(dim=2)
        nhej_p = nhej_p**self.kappa
        nhej_p = torch.div(nhej_p, nhej_p.sum(dim=1,keepdim=True))
        # combine two deletion
        overal_p = nhej_p * (1-mh_strength.view(-1,1)) + mmej_p * (mh_strength.view(-1,1))
        return overal_p

    def configure_optimizers(self):
        lr = self.hparams.lr
        betas = self.hparams.betas

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=betas)
        return optimizer

    def training_step(self, train_batch, batch_idx, step='train'):

        Feat, x, y, cmatrix = train_batch
        p_pred = self.forward(Feat)
        bs = y.shape[0]

        assert cmatrix.shape == (bs, 557, 557)

        # apply class redundant
        p_pred = torch.bmm(p_pred.squeeze(1), cmatrix).unsqueeze(1)
        y = torch.bmm(y.squeeze(1), cmatrix).unsqueeze(1)

        loss = F.cross_entropy(p_pred, y)
        kld = self.kld_fn(p_pred, y)
        self.log(f'{step}_loss', loss)
        self.log(f'f{step}_kld', kld)
        return loss

    @torch.no_grad()
    def validation_step(self, val_batch, batch_idx, step='val'):
        loss = self.training_step(val_batch, batch_idx, 'val')

    @torch.no_grad()
    def test_step(self, test_batch, batch_idx):
        loss = self.training_step(test_batch, batch_idx, 'test')
    
    @torch.no_grad()
    def predict_step(self, test_batch, batch_idx):
        Feat, x, y, cmatrix = test_batch
        p_pred = self.forward(Feat)
        return p_pred

class linear_for_Nonparams(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.mmej_regressor = nn.Sequential(
            nn.Linear(3,1,bias=False),
            nn.Softmax(dim=1)
        )
    
    def forward(self, test_batch):
        Feat, x, y, cmatrix = test_batch
        nhej_p, mmej_feat_m, mh_strength = Feat
        mmej_p = self.mmej_regressor(mmej_feat_m).squeeze(dim=2)
        nhej_p = torch.div(nhej_p, nhej_p.sum(dim=1,keepdim=True))
        # combine two deletion
        overal_p = nhej_p * (1-mh_strength.view(-1,1)) + mmej_p * (mh_strength.view(-1,1))
        return overal_p

class Deep_inDecay(deletion_decay_profiler):
    """
    The CRSIPR deletion predictor. We model the deletion events as 
    """

    def __init__(self,
                 hidden: list = [16],
                 beta: float = None,
                 kappa: float = None,
                 mu: float = None,
                 sigma: float = None,
                 update_hyper: bool = False):
        super().__init__()
        self.update_hyper = update_hyper

        for attr_n, attr in zip(
                ['kappa', 'mu', 'sigma'], [kappa, mu, sigma]
        ):
            self.float_to_params(attr_n, attr)

        dims = [3] + hidden + [1]

        mmej_mlp = [nn.Sequential(nn.Linear(din, dout, bias=True), nn.Mish())
                    for din, dout in zip(dims[:-1], dims[1:])]
        mmej_mlp += [nn.Softmax(dim=1)]
        self.mmej_regressor = nn.Sequential(*mmej_mlp)
        self.kld_fn = torchmetrics.KLDivergence()
