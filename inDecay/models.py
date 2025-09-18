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

		if isinstance(preds, torch.Tensor) and (len(target.shape)==1):
			preds = [preds.squeeze()]
			target = [target.squeeze()]

		for pred_i, target_i in zip(preds, target):

			assert pred_i.shape == target_i.shape
			non0k= len(np.where(target_i.cpu().detach().numpy() != 0)[0])
			k = self.k if self.k <= non0k else non0k

			pred_idxs = torch.topk(pred_i, k=k, dim=0).indices.cpu().numpy()
			target_idxs = torch.topk(target_i, k=k, dim=0).indices.cpu().numpy()

			batch_overlap = len(np.intersect1d(pred_idxs, target_idxs))
			# for i_p, i_t in zip(pred_idxs, target_idxs):
			#	 batch_overlap += len(np.intersect1d(i_p, i_t))
			batch_overlap_nor = batch_overlap / k

			self.overlap = self.overlap.float() + batch_overlap_nor # type: ignore
			self.total += 1  # type: ignore

	def compute(self):
		return self.overlap.float() *self.k / self.total  # type: ignore
	
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
		#	 return torch.Tensor([0])
		# else:
		r = torch.corrcoef(torch.stack([self.pred, self.target]))[0,1]
		return r**2


class Base_del_model(pl.LightningModule):
	"""
	The base pl model that takes in (x), y , c and then predict the probabity
	Args:
	T : temperature of the softmax function
	lr : learning, 
	L1_lambda : L1 regularization loss,
	L2_lambda : L2 regularization loss,
	renormalize_thres : the threshold of event probability. Event less than thres will be removed
	optim_class : str, the class of optimizer
	"""

	def __init__(self, T : float = 1.0,  lr: Union[float, int] = 3e-4, L1_lambda: float = 0, L2_lambda: float = 1e-9, renormalize_thres:float = 0.0, optim_class="Adam"):
		super().__init__()
		self.save_hyperparameters()
		self.T = T
		self.lr = lr
		self.L1_lambda = L1_lambda
		self.L2_lambda = L2_lambda
		self.train_kld = torchmetrics.KLDivergence()
		self.val_kld = torchmetrics.KLDivergence()
		self.del_regressor = nn.Identity()
		self.top1_recall = Topk_Event_Overlapping(1)
		self.top3_recall = Topk_Event_Overlapping(3)
		self.top5_recall = Topk_Event_Overlapping(5)
		self.top10_recall = Topk_Event_Overlapping(10)
		self.optim_class = optim_class
		self.renormalize_thres = renormalize_thres 
	

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

	def compute_Loss(self, out, batch, reduce='mean'):
		"""
		NLL loss 
		"""
		# TODO: weight case by total count
		#  = 200 for > 200 or S
		y = batch[1]
		if isinstance(y, list):
			cre_list = []
			for pred_i, y_i in zip(out, y):   #  iterate over each sample
				cre_list.append(
					-1* torch.multiply(torch.log(pred_i+1e-5), y_i).sum()
				)
			
			cre = torch.stack(cre_list)

		elif isinstance(y, torch.Tensor):
			cre = -1* torch.multiply(torch.log(out+1e-5), y).sum(dim=1)#.mean()

		if reduce is None:
			cre = cre
		elif reduce == 'mean':
			cre = cre.mean()
		elif reduce == 'sum':
			cre = cre.sum()
		else:
			raise ValueError("invalid reduce of NLL loss")
		return cre
	

	def compute_major(self, y, out, thre=0.25, reduction='mean'):
		"""
		Compute the recall ratio of major events (values >= thre) in y and out.
		Args:
			y: Ground truth tensor of shape (batch_size, num_events)
			out: Predicted tensor of shape (batch_size, num_events)
			thre: Threshold to consider an event as 'major'
			reduction: 'mean' returns the average recall over the batch,
					'none' returns a list of recall ratios for each sample
		Returns:
			recall: mean recall ratio or list of recall ratios
		"""
		recalls = []
		for predi, yi in zip(out, y):
			y_me_idx = (yi >= thre).nonzero(as_tuple=True)[0]
			out_me_idx = (predi >= thre).nonzero(as_tuple=True)[0]
			if len(y_me_idx) > 0:
				overlap = len(set(y_me_idx.tolist()).intersection(set(out_me_idx.tolist()))) / len(set(y_me_idx.tolist()))
			else:
				overlap = float('nan')
			recalls.append(overlap)
		if reduction == 'mean':
			# Remove nan values before averaging
			recalls = [r for r in recalls if not (isinstance(r, float) and torch.isnan(torch.tensor(r)))]
			return sum(recalls) / len(recalls) if recalls else float('nan')
		else:
			return recalls


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
		# Mask elements <0.05 and renormalize

	def mask_and_renormalize(self, pred):
		"""
		POST-processing function
		remove events that with predicted probability smaller than threshold
		"""
		if self.renormalize_thres == 0:
			return pred
		elif isinstance(pred, torch.Tensor):
			mask = pred >= self.renormalize_thres
			masked_pred = pred * mask.float()
			sum_pred = masked_pred.sum(dim=-1, keepdim=True)
			sum_pred = torch.where(sum_pred == 0, torch.ones_like(sum_pred), sum_pred)
			return masked_pred / sum_pred
		elif isinstance(pred, list):
			pred_y = []
			for pred_i in pred:
				mask = pred_i >= self.renormalize_thres
				pred_im = pred_i * mask.float()
				if np.isclose(pred_im.sum().item(), 1):
					pred_y.append(pred_im)
				else:
					pred_y.append(pred_im / pred_im.sum())
			pred= pred_y
		return pred

	def training_step(self, train_batch, batch_idx):
		
		# forward
		X, y = train_batch[:2]
		if isinstance(y, list):
			p_pred = [self.forward(x) for x in X]
		else:
			p_pred = self.forward(X)
		
		cre = self.compute_Loss(self.mask_and_renormalize(p_pred), train_batch)

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
		
		self.top1_recall(p_pred, y)
		self.top5_recall(p_pred, y)
		self.top10_recall(p_pred, y)
		# logging
		self.log('train_mse', mse, sync_dist=False, batch_size=len(y))
		self.log('train_cre', cre, sync_dist=False, batch_size=len(y))
		self.log('train_L1', L1, sync_dist=False, batch_size=len(y))
		self.log('train_L2', L2, sync_dist=False, batch_size=len(y))
		self.log('train_kld', kld, batch_size=len(y))
		self.log('train_top1recall', self.top1_recall, batch_size=len(y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log('train_top5recall', self.top5_recall, batch_size=len(y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log('train_top10recall', self.top10_recall, batch_size=len(y), on_step=False, on_epoch=True, prog_bar=True, logger=True)

		# the final loss is defined here
		loss = cre + L1*self.L1_lambda + L2*self.L2_lambda
		return loss

	def validation_step(self, train_batch, batch_idx):
		
		X, y = train_batch[:2]
		
		if isinstance(y, list):
			p_pred = [self.forward(x) for x in X]
		else:
			p_pred = self.forward(X)

		cre = self.compute_Loss(p_pred, train_batch)
		y = self.normalize_y(y)

		if isinstance(y, list):
			mse = torch.stack([F.mse_loss(pred_i, y_i) for pred_i, y_i in zip(p_pred, y)]).mean()
			kld = torch.stack([self.train_kld(p.unsqueeze(0) + 1e-14, y_i.unsqueeze(0)+ 1e-14) for p, y_i in zip(p_pred, y)]).mean()
		else:
			mse = F.mse_loss(p_pred.squeeze(), y)
			kld = self.train_kld(p_pred+1e-14, y+1e-14)
		self.top1_recall(p_pred, y)
		self.top5_recall(p_pred, y)
		self.top10_recall(p_pred, y)

		# logging
		self.log('val_mse', mse, batch_size=len(y))
		self.log('val_cre', cre, batch_size=len(y))
		self.log('val_kld', kld, batch_size=len(y))
		self.log('val_top1recall', self.top1_recall, batch_size=len(y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log('val_top5recall', self.top5_recall, batch_size=len(y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
		self.log('val_top10recall', self.top10_recall, batch_size=len(y), on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return cre

	def predict_step(self, batch, batch_idx):
		X, y = batch
		p_pred = [self.forward(x).unsqueeze(0) for x in X]
		return p_pred



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
		mask = self.get_mask(Feat_TSs[1]).squeeze()	   # b, 891
		nhej_p = self.nhej_regressor(nhej_feat_m).squeeze()
		mask_p = torch.mul(nhej_p, mask.long())
		# return  mask_p / (mask_p.sum(dim=1, keepdim=True) + 1e-6) # to stablize the divid
		return torch.softmax(mask_p, dim=1)

	def mmej_forward(self, mmej_feat_m):
		ss, mml, dl = torch.chunk(mmej_feat_m, 3, dim=-1)
		mask = self.get_mask(mml).squeeze()	   # b, 891
		# mmejpre = self.mmej_regressor(mmej_feat_m)

		mmej_p = self.mmej_regressor(mmej_feat_m).squeeze()  # b, 891
		mask_p = torch.mul(mmej_p, mask.long())
		# return mask_p / (mask_p.sum(dim=1, keepdim=True) + 1e-6)
		return torch.softmax(mask_p, dim=1)


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
		y_pred = torch.softmax(Out.squeeze(), dim=0)
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
		X, y = train_batch[:2]

		Out = self.del_regressor(x)  # [b, N_indel, 3633] -> [b, N_indel,1]
		y_pred = torch.softmax(Out.squeeze(2), dim=1)

		scaler_v = self.get_scaler(x)
		scaled_y = torch.multiply(y_pred, scaler_v)

		return scaled_y / scaled_y.sum(), y


class ST_DeepDecay(Base_del_model):
	"""
	inDecay's MLP model
	"""
	def __init__(self, inputsize=9, outputsize=1, hidden=[16], lr=3e-4, L1_lambda=3e-4, L2_lambda=3e-4, T=1, renormalize_thres=0):
		super().__init__(lr=lr, L1_lambda=L1_lambda, L2_lambda=L2_lambda, T=T, renormalize_thres=renormalize_thres)
		self.lr = lr
		self.T = T
		
		layer_size = [inputsize] + hidden 
		layer_ls = [nn.Sequential(nn.Linear(din, dout, bias=True), nn.Mish())
				for din, dout in zip(layer_size[:-1] , layer_size[1:])]
		layer_ls += [nn.Linear(hidden[-1],1)]
		self.del_regressor = nn.Sequential(*layer_ls)


		self.l1_crit = nn.L1Loss(size_average=False) 
		self.train_kld = torchmetrics.KLDivergence()
		self.val_kld = torchmetrics.KLDivergence()
	
	def forward(self, x):
		Out = self.del_regressor(x)/self.T # [b, N_indel, 3633] -> [b, N_indel,1]
		y_pred = torch.softmax(Out.squeeze(), dim=0)
		return y_pred



class ST_DeepDecay_dropout(ST_DeepDecay):
	"""
	repeat Lindel's linear model
	"""
	def __init__(self, inputsize=9, outputsize=1, hidden=[16], lr=3e-4, L1_lambda=3e-4, L2_lambda=3e-4, T=1, renormalize_thres=0.0):
		super().__init__(inputsize=inputsize, outputsize=outputsize, hidden=hidden, lr=lr,L1_lambda=L1_lambda,L2_lambda=L2_lambda, T=T, renormalize_thres=renormalize_thres)

		layer_size = [inputsize] + hidden 

		layer_ls = [nn.Sequential(nn.Linear(din, dout, bias=True), nn.Mish(), nn.Dropout(p=0.3))
				for din, dout in zip(layer_size[:-1] , layer_size[1:])]
		layer_ls += [nn.Linear(hidden[-1],1)]
		self.del_regressor = nn.Sequential(*layer_ls)

class ST_DeepDecay_mul(ST_DeepDecay):
	"""
	repeat model
	"""
	def __init__(self, inputsize=9, outputsize=1, hidden=[16], lr=3e-4, L1_lambda=3e-4, L2_lambda=3e-4, T=1, renormalize_thres=0.0):
		super().__init__(inputsize=inputsize, outputsize=outputsize, hidden=hidden, lr=lr,L1_lambda=L1_lambda,L2_lambda=L2_lambda, T=T, renormalize_thres=renormalize_thres)

	def compute_Loss(self, out, batch, reduce='mean'):
		"""
		Multi-nomial loss 
		"""
		y = batch[1]
		with torch.no_grad():
			total_count = torch.tensor([yi.sum() for yi in y]).to(y[0].device)
			count= torch.clamp(total_count/100, min=1.0)
			# weight /= weight.sum()
		if isinstance(y, list):
			cre_list = []
			for pred_i, y_i, count_i in zip(out, y, count):   #  iterate over each sample
				cre_list.append(-1* torch.multiply(torch.log(pred_i+1e-5),torch.div(y_i, count_i)).sum())
			cre = torch.stack(cre_list)

		elif isinstance(y, torch.Tensor):
			cre = -1* torch.multiply(torch.log(out+1e-5),torch.div(y, count)).sum()
		if reduce is None:
			cre = cre
		elif reduce == 'mean':
			cre = cre.mean()
		elif reduce == 'sum':
			cre = cre.sum()
		else:
			raise ValueError("invalid reduce of NLL loss")
		return cre

class ST_DeepDecay_r2(ST_DeepDecay):
	"""
	reweight sample loss based on decodR/ICE r2 score
	"""
	def __init__(self, inputsize=9, outputsize=1, hidden=[16], lr=3e-4, L1_lambda=3e-4, L2_lambda=3e-4, T=1, renormalize_thres=0.05):
		super().__init__(inputsize=inputsize, outputsize=outputsize, hidden=hidden, lr=lr,L1_lambda=L1_lambda,L2_lambda=L2_lambda, T=T, renormalize_thres=renormalize_thres)

	def compute_Loss(self, out, batch, reduce='mean'):
		"""
		Multi-nomial loss 
		"""

		y = batch[1]  
		r2 = batch[2]  
		with torch.no_grad():
			total_count = torch.tensor([yi.sum() for yi in y]).to(y[0].device)
			count= torch.clamp(total_count/100, min=1.0)
			# weight /= weight.sum()
		if isinstance(y, list):
			cre_list = []
			for pred_i, y_i, count_i, r2_i in zip(out, y, count, r2):   #  iterate over each sample
				cre_list.append(
					-1 * r2_i * torch.multiply(torch.log(pred_i+1e-5), y_i).sum() # neg log likelihood
					#-1 * r2_i * torch.log(pred_i+1e-5).sum()
					)
			cre = torch.stack(cre_list)

		elif isinstance(y, torch.Tensor):
			cre = torch.sum(-1* torch.multiply(torch.log(out+1e-5),torch.Tensor(r2).to(y.device) ))
		if reduce is None:
			cre = cre
		elif reduce == 'mean':
			cre = cre.mean()
		elif reduce == 'sum':
			cre = cre.sum()
		else:
			raise ValueError("invalid reduce of NLL loss")
		return cre

class ST_DeepDecay_weight(ST_DeepDecay_dropout):
	"""
	repeat model
	"""
	def __init__(self, inputsize=9, outputsize=1, hidden=[16], lr=3e-4, L1_lambda=3e-4, L2_lambda=3e-4):
		super().__init__(inputsize=inputsize, outputsize=outputsize, hidden=hidden, lr=lr,L1_lambda=L1_lambda,L2_lambda=L2_lambda)
	
	def compute_Loss(self, out, batch):
		"""
		Multi-nomial loss 
		"""
		y = batch[1]
		with torch.no_grad():
			total_count = torch.tensor([yi.sum() for yi in y]).to(y[0].device)
			
			weight = torch.clamp(total_count/1000, min=0.3, max=1.0)
			# weight /= weight.sum()
		
		# redunction set to None to return the cre of each sample
		y_norm = self.normalize_y(y)
		cre = super().compute_Loss(out, y_norm, reduce=None)
		loss = torch.multiply(cre, weight).sum() / weight.sum()
		return loss 
	
class ST_DeepDecay_Multinomial(ST_DeepDecay_dropout):
	"""
	DeepDecay with Multinomial loss
	"""
	def __init__(self, inputsize=9, outputsize=1, hidden=[16], lr=3e-4, L1_lambda=3e-4, L2_lambda=3e-4, T=1):
		super().__init__(inputsize=inputsize, outputsize=outputsize, hidden=hidden, lr=lr,L1_lambda=L1_lambda,L2_lambda=L2_lambda)
	def compute_Loss(self, out, batch):
		"""
		Multi-nomial loss 
		"""
		y = batch[1]
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
		cre = self.compute_Loss(p_pred, train_batch)	

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
		# cre = self.compute_Loss(p_pred, train_batch)	
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
	
