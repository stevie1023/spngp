import numpy as np 
from learnspngp import Mixture, Separator, GPMixture, Color
import gc
import torch
import gpytorch
from gpytorch.kernels import *
from gpytorch.likelihoods import *
from gpytorch.mlls import *
from torch.optim import *
from torch.utils.data import TensorDataset, DataLoader
# from scipy.optimize import minimize
# from numpy.linalg   import inv, cholesky

class Sum:
    def __init__(self, **kwargs):
        self.children = []
        self.weights  = []
        return None

    def forward(self, x_pred, label_y, **kwargs):
        if len(x_pred) == 0:
            return np.zeros(1), np.zeros(1)

        # Sometimes Sums only have one child 
        # (when using 1GP for instance) therefore 
        # we return fast:
        if len(self.children) == 1:
            r_ = self.children[0].forward(x_pred, label_y,**kwargs)
            return r_[0].reshape(-1, 1), r_[1].reshape(-1, 1), r_[2].reshape(-1, 1)



        if len(self.children) == 2:
            _wei = np.array(self.weights).reshape(-1, 1)
            mu_x = np.empty((len(x_pred), len(self.weights)))
            co_x = np.empty((len(x_pred), len(self.weights)))
            mll_x = np.empty((len(x_pred), len(self.weights)))
            r1_ = self.children[0].forward(x_pred,label_y,**kwargs)
            r2_ = self.children[1].forward(x_pred,label_y, **kwargs)
            mu_x[:, 0], co_x[:, 0], mll_x[:, 0] = r1_[0].squeeze(-1), r1_[1].squeeze(-1), r1_[2].squeeze(-1)
            mu_x[:, 1], co_x[:, 1], mll_x[:,1] = r2_[0].squeeze(-1), r2_[1].squeeze(-1),r2_[2].squeeze(-1)

            # co_x1 = np.empty((len(x_pred), len(self.weights)))
            # co_x2 = np.empty((len(x_pred), len(self.weights)))
            # co_x1[:, 0] = co_x[:, 0]
            # co_x2[:, 1] = co_x[:, 1]
            # # index_min2 = np.argmin((np.mean(co_x1), np.mean(co_x2)))
            # mll_x1 = np.empty((len(x_pred), len(self.weights)))
            # mll_x2 = np.empty((len(x_pred), len(self.weights)))
            # mll_x1[:, 0] = mll_x[:, 0]
            # mll_x2[:, 1] = mll_x[:, 1]
            return_mu, return_cov, mll_x_= mu_x @ _wei, co_x @_wei, mll_x @ _wei
            # index_max_mll = np.argmax((np.mean(mll_x1 @_wei), np.mean(mll_x2 @_wei)))
            # return_mu, return_cov = mu_x[:,index_max_mll].reshape(-1, 1), co_x[:, index_max_mll].reshape(-1,1)
            return return_mu, return_cov, mll_x_

        if len(self.children) == 3:
            _wei = np.array(self.weights).reshape(-1, 1)
            mu_x = np.empty((len(x_pred), len(self.weights)))
            co_x = np.empty((len(x_pred), len(self.weights)))
            mll_x = np.empty((len(x_pred), len(self.weights)))
            for i, child in enumerate(self.children):
                mu_c, co_c, mll_x1 = child.forward(x_pred,label_y, **kwargs)
                mu_x[:, i], co_x[:, i], mll_x[:, i] = mu_c.squeeze(-1), co_c.squeeze(-1), np.squeeze(mll_x1)

            # # #
            # if self.children[0] is GPMixture:
            #     return_mu, return_cov = mu_x @ _wei, co_x @ _wei
            #     return return_mu, return_cov, mll_x @ _wei
            # else:
            # co_x1 = np.empty((len(x_pred), len(self.weights)))
            # co_x2 = np.empty((len(x_pred), len(self.weights)))
            # co_x3 = np.empty((len(x_pred), len(self.weights)))
            #
            #
            # co_x1[:, 0] = co_x[:, 0]
            # co_x2[:, 1] = co_x[:, 1]
            # co_x3[:, 2] = co_x[:, 2]
            # #
            #     #
            # # index_min2 = np.argmin((np.mean(co_x1), np.mean(co_x2), np.mean(co_x3)))
                # return_mu, return_cov, mll_x_= mu_x @ _wei, co_x[:,index_min2].reshape(-1, 1), mll_x @_wei
            # # if self.children[0] is GPMixture:
            # mll_x1 = np.empty((len(x_pred), len(self.weights)))
            # mll_x2 = np.empty((len(x_pred), len(self.weights)))
            # mll_x3 = np.empty((len(x_pred), len(self.weights)))
            # mll_x1[:, 0] = mll_x[:, 0]
            # mll_x2[:, 1] = mll_x[:, 1]
            # mll_x3[:, 2] = mll_x[:, 2]
            # index_max_mll = np.argmax((np.mean(mll_x1 @_wei), np.mean(mll_x2 @_wei), np.mean(mll_x3 @_wei)))
            # return_mu, return_cov = mu_x[:, index_max_mll].reshape(-1, 1), co_x[:, index_max_mll].reshape(-1, 1)
            return_mu, return_cov = mu_x @_wei, co_x @_wei
            return return_mu, return_cov, mll_x @_wei

            # else:
            # return_mu, return_cov = mu_x @ _wei, co_x @ _wei
            # return return_mu, return_cov, mll_x @_wei


    def true_variance(self, x_pred, mu, **kwargs):
        if len(self.children) == 1:
            r_ = self.children[0].true_variance(x_pred, mu, **kwargs)
            return r_[0].reshape(-1, 1), r_[1].reshape(-1, 1)

        _wei = np.array(self.weights).reshape(-1, 1)
        mu_x = np.empty((len(mu), len(self.weights)))
        co_x = np.empty((len(mu), len(self.weights)))

        for i, child in enumerate(self.children):
            mu_c, co_c = child.true_variance(x_pred, mu, **kwargs)
            mu_x[:, i], co_x[:, i] = mu_c.squeeze(-1), co_c.squeeze(-1)

        return mu_x @ _wei, ((mu_x-mu)**2 + co_x) @ _wei

    def update(self):
        c_mllh       = np.array([c.update() for c in self.children])
        new_weights  = np.exp(c_mllh)
        self.weights = new_weights / np.sum(new_weights)

        # w_prior      = np.log(self.weights)
        # c_mllh       = np.array([c.update() for c in self.children])
        # w_post       = np.exp(w_prior + c_mllh)
        # z            = np.sum(np.exp(w_post))
        # self.weights = w_post / z
        return np.log(np.sum(new_weights))

    def __repr__(self, level = 0, **kwargs):
        _wei = dict.get(kwargs, 'extra')
        _wei = Color.flt(_wei) if _wei else ""
        _sel = " "*(level) + f"{_wei} ✚ Sum"
        for i, child in enumerate(self.children):
            _wei = self.weights[i]
            _sel += f"\n{child.__repr__(level+2, extra=_wei)}"
        
        return f"{_sel}"

class Split:
    def __init__(self, **kwargs):
        self.children = []
        self.split     = kwargs['split']
        self.dimension = kwargs['dimension']
        self.depth     = kwargs['depth']
        return None

    def forward(self, x_pred,label_y, **kwargs):
        smudge = dict.get(kwargs, 'smudge', 0)

        mu_x = np.zeros((len(x_pred), 1))
        co_x = np.zeros((len(x_pred), 1))
        mll_x = np.zeros((len(x_pred), 1))
        
        left, right = self.children[0], self.children[1]
        
        smudge_scales = dict.get(kwargs,   'smudge_scales', {})
        smudge_scale  = dict.get(smudge_scales, self.depth,  1) 

        left_idx  = np.where(x_pred[:, self.dimension] <= (self.split + smudge*smudge_scale))[0]
        right_idx = np.where(x_pred[:, self.dimension] >  (self.split - smudge*smudge_scale))[0]
        
        if not smudge:
            mu_x[left_idx],  co_x[left_idx], mll_x[left_idx] = left.forward(x_pred[left_idx],label_y[left_idx], **kwargs)
            mu_x[right_idx], co_x[right_idx], mll_x[right_idx] = right.forward(x_pred[right_idx],label_y[right_idx], **kwargs)
        else:
            inter_idx = np.intersect1d(left_idx, right_idx)

            mu_x[left_idx],  co_x[left_idx], mll_x[left_idx] = left.forward(x_pred[left_idx],label_y[left_idx], **kwargs)
            left_mu, left_co, left_mll                = mu_x[inter_idx], co_x[inter_idx], mll_x[inter_idx]

            mu_x[right_idx], co_x[right_idx], mll_x[right_idx] = right.forward(x_pred[right_idx],label_y[right_idx], **kwargs)
            right_mu, right_co, right_mll               = mu_x[inter_idx], co_x[inter_idx], mll_x[inter_idx]
    
            ss = (np.sqrt(left_co)**-2 + np.sqrt(right_co)**-2)**-1
            mu = ss*(left_mu/left_co + right_mu/right_co)
            #modified
            mll_inter = ss * (left_mll / left_co + right_mll / right_co)

            mu_x[inter_idx], co_x[inter_idx], mll_x[inter_idx]= mu, ss, mll_inter
        
        return mu_x, co_x, mll_x
    
    def update(self):
        return np.sum([c.update() for c in self.children])
    
    def true_variance(self, x_pred, label_y,mu, **kwargs):
        smudge = dict.get(kwargs, 'smudge', 0)

        mu_x = np.zeros((len(x_pred), 1))
        co_x = np.zeros((len(x_pred), 1))
        mll_x = np.zeros((len(x_pred), 1))

        left, right = self.children[0], self.children[1]
        
        smudge_scales = dict.get(kwargs,   'smudge_scales', {})
        smudge_scale  = dict.get(smudge_scales, self.depth,  1) 

        left_idx  = np.where(x_pred[:, self.dimension] <=  (self.split + smudge*smudge_scale))[0]
        right_idx = np.where(x_pred[:, self.dimension]  >  (self.split - smudge*smudge_scale))[0]
        
        if not smudge:
            mu_x[left_idx],  co_x[left_idx], mll_x[left_idx] = left.true_variance(x_pred[left_idx],label_y[left_idx], mu, **kwargs)
            mu_x[right_idx], co_x[right_idx], mll_x[right_idx]= right.true_variance(x_pred[right_idx], label_y[right_idx],mu, **kwargs)
        else:
            inter_idx = np.intersect1d(left_idx, right_idx)

            mu_x[left_idx],  co_x[left_idx], mll_x[left_idx]  = left.true_variance(x_pred[left_idx], label_y[left_idx],mu, **kwargs)
            left_mu, left_co,left_mll                 = mu_x[inter_idx], co_x[inter_idx],mll_x[inter_idx]

            mu_x[right_idx], co_x[right_idx], mll_x[right_idx] = right.true_variance(x_pred[right_idx],label_y[right_idx], mu, **kwargs)
            right_mu, right_co , right_mll            = mu_x[inter_idx], co_x[inter_idx], mll_x[inter_idx]
    
            ss = (np.sqrt(left_co)**-2 + np.sqrt(right_co)**-2)**-1
            mu = ss*(left_mu/left_co + right_mu/right_co)
            mll_inter = ss * (left_mll / left_co + right_mll / right_co)

            mu_x[inter_idx], co_x[inter_idx], mll_x[inter_idx] = mu, ss, mll_inter
        
        return mu_x, co_x, mll_x

    def __repr__(self, level = 0, **kwargs):
        _wei = dict.get(kwargs, 'extra')
        _wei = Color.flt(_wei) if _wei else ""
        _spl = Color.val(self.split, f=1, color='yellow')
        _dim = Color.val(self.dimension, f=0, color='orange', extra="dim=")
        _dep = Color.val(self.depth, f=0, color='blue', extra="dep=")
        _sel = " "*(level-1) + f"{_wei} ⓛ Split {_spl} {_dim} {_dep}"
        for child in self.children:
            _sel += f"\n{child.__repr__(level+10)}"
        
        return f"{_sel}"

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, **kwargs):
        x, y       = kwargs['x'], kwargs['y']


        likelihood = kwargs['likelihood']
        gp_type    = kwargs['type']
        xd         = x.shape[1]
        
        active_dims = torch.tensor(list(range(xd)))

        super(ExactGPModel, self).__init__((x), (y), likelihood)


        self.mean_module = gpytorch.means.ConstantMean()
        
        if gp_type == 'mixed':
            m  = MaternKernel(nu=0.5, ard_num_dims=xd, active_dims=active_dims)
            l  = LinearKernel()
            self.covar_module = ScaleKernel(m + l)
            return
        elif gp_type == 'matern0.5':
            k = MaternKernel(nu=0.5)
        elif gp_type == 'matern1.5':
            k = MaternKernel(nu=1.5)
        elif gp_type == 'matern2.5':
            k = MaternKernel(nu=2.5)
        elif gp_type == 'rbf':
            k = RBFKernel()
        elif gp_type == 'matern0.5_ard':
            k = MaternKernel(nu=0.5, ard_num_dims=xd, active_dims=active_dims)
        elif gp_type == 'matern1.5_ard':
            k = MaternKernel(nu=1.5, ard_num_dims=xd, active_dims=active_dims)
        elif gp_type == 'matern2.5_ard':
            k = MaternKernel(nu=2.5, ard_num_dims=xd, active_dims=active_dims)
        elif gp_type == 'rbf_ard':
            k = RBFKernel(ard_num_dims=xd)
        elif gp_type == 'linear':
            k = LinearKernel(ard_num_dims=xd) # ard_num_dims for linear doesn't actually work
        else:
            raise Exception("Unknown GP type")

        k3 = PolynomialKernel(2)
        k2 = SpectralMixtureKernel(num_mixtures=4,  ard_num_dims=xd, active_dims=active_dims )
        # k1 = GridInterpolationKernel(gpytorch.kernels.RBFKernel(), grid_size=gpytorch.utils.grid.choose_grid_size(x), num_dims=4)
        # self.covar_module = ScaleKernel(k2)
        self.covar_module = k2

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#
class ExactGPModel1(gpytorch.models.ExactGP):
    def __init__(self, **kwargs):
        x, y = kwargs['x'], kwargs['y']
        likelihood = kwargs['likelihood']
        gp_type = kwargs['type']
        xd = x.shape[1]

        active_dims = torch.tensor(list(range(xd)))

        super(ExactGPModel1, self).__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        k1 = LinearKernel()
        k2 = RBFKernel()
        k3 = MaternKernel(nu=1.5)
        k4 = RQKernel()
        self.covar_module = ScaleKernel(k2+k4)



    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPModel2(gpytorch.models.ExactGP):
    def __init__(self, **kwargs):
        x, y = kwargs['x'], kwargs['y']
        likelihood = kwargs['likelihood']
        gp_type = kwargs['type']
        xd = x.shape[1]

        active_dims = torch.tensor(list(range(xd)))

        super(ExactGPModel2, self).__init__(x, y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()


        # gp_type == 'rbf_ard':
        # k = RBFKernel(ard_num_dims=xd)
        # gp_type == 'linear':
        k1 = RBFKernel()
        k2 = LinearKernel()  # ard_num_dims for linear doesn't actually work
        k3 = PolynomialKernel(2)

        self.covar_module = ScaleKernel(k2+k1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GP:
    def __init__(self, **kwargs):
        self.type = kwargs['type']
        self.mins = kwargs['mins']
        self.maxs = kwargs['maxs']
        self.mll  = 0
        self.x    = dict.get(kwargs, 'x', [])
        self.y    = dict.get(kwargs, 'y', [])
        self.n    = None
    
    def forward(self, x_pred,label_y, **kwargs):
        ##modified
        mu_gp, co_gp , mll= self.predict(x_pred,label_y)

        # if dict.get(kwargs, 'store_results'):
        #     self.stored_mu, self.stored_cov = mu_gp, co_gp

        return mu_gp, co_gp, mll
    
    def true_variance(self, x_pred, mu, **kwargs):
        mu_gp, co_gp = self.predict(x_pred) 
        return mu_gp, co_gp 
        #return self.stored_mu, self.stored_cov
        
    def update(self):
        return self.mll #np.log(self.n)*0.01 #-self.mll - 1/np.log(self.n)

    def predict(self, X_s, label_y):
        self.model.eval()
        self.likelihood.eval()
        x = torch.from_numpy(X_s).float() #.to('cpu') #.cuda()
        y = torch.from_numpy(label_y.ravel()).float()

        test_dataset = TensorDataset(x, y)
        test_loader =  DataLoader(test_dataset)
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(x))
            pm, pv = observed_pred.mean, observed_pred.variance
            lls=[]
            ##modified
            for x_batch, y_batch in test_loader:
                lls.append(self.model.likelihood.log_marginal(y_batch, self.model(x_batch)).item())
            # mll_ = ExactMarginalLogLikelihood(self.likelihood, self.model)

            # mll = mll_(observed_pred, y).item()

        x.detach()

        del x

        gc.collect()
        #modified
        return pm.detach().cpu(), pv.detach().cpu(), np.array(lls)
    
    def init(self, **kwargs):
        lr    = dict.get(kwargs, 'lr',    0.20)
        steps = dict.get(kwargs, 'steps',  100)
        
        self.n      = len(self.x)
        self.cuda   = dict.get(kwargs, 'cuda') and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        if self.cuda: 
            torch.cuda.empty_cache()
        
        self.x = torch.from_numpy(self.x).float().to(self.device)

        self.y = torch.from_numpy(self.y.ravel()).float().to(self.device)

        
        #noises = torch.ones(self.n) * 1
        #self.likelihood = FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=True)
        self.likelihood = GaussianLikelihood()
        # noise_constraint=gpytorch.constraints.LessThan(1e-2))
        # (noise_prior=gpytorch.priors.NormalPrior(3, 20))
        self.model = ExactGPModel(x = self.x, y = self.y, likelihood = self.likelihood, type = self.type).to(self.device)#.cuda()
        self.optimizer = Adam([{'params': self.model.parameters()}], lr=lr)
        self.model.train()
        self.likelihood.train()

        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

        print(f"\tGP {self.type} init completed. Training on {self.device}")
        for i in range(steps):
            self.optimizer.zero_grad()   # Zero gradients from previous iteration
            output = self.model(self.x)  # Output from model
            loss = -mll(output, self.y)  # Calc loss and backprop gradients
            if i > 0 and i % 10 ==0:
                print(f"\t Step {i+1}/{steps}, -mll(loss): {round(loss.item(), 3)}")

            loss.backward()
            self.optimizer.step()
        
        # LOG LIKELIHOOD NOW POSITIVE
        self.mll = -loss.detach().item()

        print(f"\tCompleted. +mll: {round(self.mll,3)}")

        self.x.detach()
        self.y.detach()
              
        #del self.likelihood
        #self.likelihood = None
        del self.x
        del self.y
        self.x = self.y = None 
    
        self.model = self.model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()


    def init1(self, **kwargs):
        lr    = dict.get(kwargs, 'lr',    0.20)
        steps = dict.get(kwargs, 'steps',  100)

        self.n      = len(self.x)
        self.cuda   = dict.get(kwargs, 'cuda') and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        if self.cuda:
            torch.cuda.empty_cache()

        self.x = torch.from_numpy(self.x).float().to(self.device)

        self.y = torch.from_numpy(self.y.ravel()).float().to(self.device)


        #noises = torch.ones(self.n) * 1
        #self.likelihood = FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=True)
        self.likelihood = GaussianLikelihood()
        # noise_constraint=gpytorch.constraints.LessThan(1e-2))
        # (noise_prior=gpytorch.priors.NormalPrior(3, 20))
        self.model = ExactGPModel1(x = self.x, y = self.y, likelihood = self.likelihood, type = self.type).to(self.device)#.cuda()
        self.optimizer = Adam([{'params': self.model.parameters()}], lr=lr)
        self.model.train()
        self.likelihood.train()

        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        print(f"\tGP {self.type} init completed. Training on {self.device}")
        for i in range(steps):
            self.optimizer.zero_grad()   # Zero gradients from previous iteration
            output = self.model(self.x)  # Output from model
            loss = -mll(output, self.y)  # Calc loss and backprop gradients
            if i > 0 and i % 10 ==0:
                print(f"\t Step {i+1}/{steps}, -mll(loss): {round(loss.item(), 3)}")

            loss.backward()
            self.optimizer.step()

        # LOG LIKELIHOOD NOW POSITIVE
        self.mll = -loss.detach().item()

        print(f"\tCompleted. +mll: {round(self.mll,3)}")

        self.x.detach()
        self.y.detach()

        #del self.likelihood
        #self.likelihood = None
        del self.x
        del self.y
        self.x = self.y = None

        self.model = self.model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()

    def init2(self, **kwargs):
        lr = dict.get(kwargs, 'lr', 0.20)
        steps = dict.get(kwargs, 'steps', 100)

        self.n = len(self.x)
        self.cuda = dict.get(kwargs, 'cuda') and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.cuda else "cpu")
        if self.cuda:
            torch.cuda.empty_cache()

        self.x = torch.from_numpy(self.x).float().to(self.device)

        self.y = torch.from_numpy(self.y.ravel()).float().to(self.device)

        # noises = torch.ones(self.n) * 1
        # self.likelihood = FixedNoiseGaussianLikelihood(noise=noises, learn_additional_noise=True)
        self.likelihood = GaussianLikelihood()
        # noise_constraint=gpytorch.constraints.LessThan(1e-2))
        # (noise_prior=gpytorch.priors.NormalPrior(3, 20))
        self.model = ExactGPModel2(x=self.x, y=self.y, likelihood=self.likelihood, type=self.type).to(
            self.device)  # .cuda()
        self.optimizer = Adam([{'params': self.model.parameters()}], lr=lr)
        self.model.train()
        self.likelihood.train()

        mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        print(f"\tGP {self.type} init completed. Training on {self.device}")
        for i in range(steps):
            self.optimizer.zero_grad()  # Zero gradients from previous iteration
            output = self.model(self.x)  # Output from model
            loss = -mll(output, self.y)  # Calc loss and backprop gradients
            if i > 0 and i % 10 == 0:
                print(f"\t Step {i + 1}/{steps}, -mll(loss): {round(loss.item(), 3)}")

            loss.backward()
            self.optimizer.step()

        # LOG LIKELIHOOD NOW POSITIVE
        self.mll = -loss.detach().item()

        print(f"\tCompleted. +mll: {round(self.mll, 3)}")

        self.x.detach()
        self.y.detach()

        # del self.likelihood
        # self.likelihood = None
        del self.x
        del self.y
        self.x = self.y = None

        self.model = self.model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()


    def __repr__(self, level = 0, **kwargs):
        _wei = dict.get(kwargs, 'extra')
        _wei = Color.flt(_wei) if _wei else ""

        _rou = 2
        _rng = [f"{round(self.mins[i],_rou)}-{round(self.maxs[i],_rou)}" for i, _ in enumerate(self.mins)]
        
        if self.n is not None:
            _cnt = Color.val(self.n, f=0, color='green', extra="n=")
        else:
            _cnt = 0

        _mll = Color.val(self.mll, f=3, color='orange', extra="mllh=")
        return " "*(level) + f"{_wei} ⚄ GP ({self.type}) {_rng} {_cnt} {_mll}"

def structure(root_region, **kwargs):
    root = Sum()
    to_process, gps = [(root_region, root)], dict()

    gps1 = dict()  #modified
    gps2 = dict()

    gp_types = dict.get(kwargs, 'gp_types', ['mixed'])

    while len(to_process):
        gro, sto = to_process.pop() # gro = graph region object
                                    # sto = structure object
        
        if type(gro) is Mixture:
            for child in gro.children:
                if type(child) == Separator:
                    _child = Split(split=child.split, depth=child.depth, dimension=child.dimension)
                elif type(child) == Mixture:
                    _child = Sum()
                else:
                    _child = Sum()
                sto.children.append(_child)
            _cn = len(sto.children)
            sto.weights = np.ones(_cn) / _cn
            to_process.extend(zip(gro.children, sto.children))#children of gro and sto are grouped together and put into root_region
        elif type(gro) is Separator: # sto is Split
            for child in gro.children:
                sto.children.append(Sum())
            to_process.extend(zip(gro.children, sto.children))
        elif type(gro) is GPMixture: # sto is Sum
            for gp_type in gp_types:
                key = (*gro.mins, *gro.maxs, gp_type)
                if not dict.get(gps, key):
                    gps[key] = GP(type=gp_type, mins=gro.mins, maxs=gro.maxs)
                else:
                    pass
                sto.children.extend([gps[key], gps[key], gps[key]])
        elif type(gro) is tuple:  #modified
            for gp_type in gp_types:
                key = (*gro[0].mins, *gro[0].maxs, gp_type)
                key1 = (*gro[1].mins, *gro[1].maxs, gp_type)
                key2 = (*gro[2].mins, *gro[2].maxs, gp_type)
                if not dict.get(gps, key):
                    gps[key] = GP(type=gp_type, mins=gro[0].mins, maxs=gro[0].maxs)
                    gps1[key1] = GP(type=gp_type, mins=gro[1].mins, maxs=gro[1].maxs)
                    gps2[key2] = GP(type=gp_type, mins=gro[2].mins, maxs=gro[2].maxs)

                else:
                    pass
                sto.children.extend([gps[key], gps1[key1], gps2[key2]])


            _cn = len(sto.children)
            sto.weights = np.ones(_cn) / _cn
    
    return root, list(gps.values()), list(gps1.values()), list(gps2.values())  #modified

def pretty_size(size):
	"""Pretty prints a torch.Size object"""
	assert(isinstance(size, torch.Size))
	return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):
	"""Prints a list of the Tensors being tracked by the garbage collector."""
	import gc
	total_size = 0
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj):
				if not gpu_only or obj.is_cuda:
					print("%s:%s%s %s" % (type(obj).__name__, 
										  " GPU" if obj.is_cuda else "",
										  " pinned" if obj.is_pinned else "",
										  pretty_size(obj.size())))
					total_size += obj.numel()
			elif hasattr(obj, "data") and torch.is_tensor(obj.data):
				if not gpu_only or obj.is_cuda:
					print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
												   type(obj.data).__name__, 
												   " GPU" if obj.is_cuda else "",
												   " pinned" if obj.data.is_pinned else "",
												   " grad" if obj.requires_grad else "", 
												   " volatile" if obj.volatile else "",
												   pretty_size(obj.data.size())))
					total_size += obj.data.numel()
		except Exception as e:
			pass        
	print("Total size:", total_size)