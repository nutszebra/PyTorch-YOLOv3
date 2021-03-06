import math
import numpy as np
import torch.optim as optim


class FakeOptimizer(object):

    def __call__(self, i):
        pass

    def step(self):
        pass

    def zero_grad(self):
        pass

    def info(self):
        pass


class Adam(object):

    def __init__(self, model, lr=0.001, betas=(0.9, 0.999), schedule=[100, 150], lr_decay=0.1, weight_decay=1.0e-4):
        self.model, self.lr, self.betas = model, lr, betas
        self.schedule, self.lr_decay, self.weight_decay = schedule, lr_decay, weight_decay
        self.optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

    def __call__(self, i):
        if i in self.schedule:
            for p in self.optimizer.param_groups:
                previous_lr = p['lr']
                new_lr = p['lr'] * self.lr_decay
                p['lr'] = new_lr
            print('{}->{}'.format(previous_lr, new_lr))
            self.lr = new_lr
            self.info()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def info(self):
        write('Optimizer')
        keys = self.__dict__.keys()
        for key in keys:
            if key == 'model':
                continue
            else:
                write('    {}: {}'.format(key, self.__dict__[key]))


class MomentumSGDNoWeightDecayOnBias(object):

    def __init__(self, model, lr, momentum, schedule=[100, 150], lr_decay=0.1, weight_decay=1.0e-4):
        self.model, self.lr, self.momentum = model, lr, momentum
        self.schedule, self.lr_decay, self.weight_decay = schedule, lr_decay, weight_decay
        params = self.no_weight_decay_on_bias(model, weight_decay)
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    @staticmethod
    def no_weight_decay_on_bias(model, weight_decay):
        params_dict = dict(model.named_parameters())
        params = []
        for key, value in params_dict.items():
            if key[-4:] == 'bias' and (('fc' in key) or ('linear' in key) or ('bn' in key)):
                params += [{'params': value, 'weight_decay': 0.0}]
            else:
                params += [{'params': value, 'weight_decay': weight_decay}]
        return params

    def __call__(self, i):
        if i in self.schedule:
            for p in self.optimizer.param_groups:
                previous_lr = p['lr']
                new_lr = p['lr'] * self.lr_decay
                print('{}->{}'.format(previous_lr, new_lr))
                p['lr'] = new_lr
            self.lr = new_lr
            self.info()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def info(self):
        write('Optimizer')
        keys = self.__dict__.keys()
        for key in keys:
            if key == 'model':
                continue
            else:
                write('    {}: {}'.format(key, self.__dict__[key]))


class MomentumSGD(object):

    def __init__(self, model, lr, momentum, schedule=[100, 150], lr_decay=0.1, weight_decay=1.0e-4):
        self.model, self.lr, self.momentum = model, lr, momentum
        self.schedule, self.lr_decay, self.weight_decay = schedule, lr_decay, weight_decay
        self.optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=True)

    def __call__(self, i):
        if i in self.schedule:
            for p in self.optimizer.param_groups:
                previous_lr = p['lr']
                new_lr = p['lr'] * self.lr_decay
                print('{}->{}'.format(previous_lr, new_lr))
                p['lr'] = new_lr
            self.lr = new_lr
            self.info()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def info(self):
        write('Optimizer')
        keys = self.__dict__.keys()
        for key in keys:
            if key == 'model':
                continue
            else:
                write('    {}: {}'.format(key, self.__dict__[key]))


class AdamW(object):

    def __init__(self, model, lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.025 * math.sqrt(64 / 50000), t_i=200):
        self.model, self.lr, self.betas = model, lr, betas
        self.eps, self.weight_decay, self.t_i = eps, weight_decay, t_i
        self.optimizer = adamw.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.eta = 1.0
        self.now = {'weight_decay': weight_decay, 'lr': lr}

    @staticmethod
    def cos(t_cur, t_i):
        return 0.5 + 0.5 * math.cos(math.pi * t_cur / t_i)

    def __call__(self, i):
        # updat eta every epoch
        self.eta = self.cos(i, self.t_i)
        for p in self.optimizer.param_groups:
            # lr
            p['lr'] = self.eta * self.lr
            # weight decay
            p['weight_decay'] = self.eta * self.weight_decay * math.sqrt(1.0 / (i + 1))
            self.now['weight_decay'], self.now['lr'] = p['weight_decay'], p['lr']
        self.info()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def info(self):
        write('Optimizer: AdamW')
        keys = self.__dict__.keys()
        for key in keys:
            if key == 'model':
                continue
            else:
                write('    {}: {}'.format(key, self.__dict__[key]))


class AMSGradW(object):

    def __init__(self, model, lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.025 * math.sqrt(64 / 50000), t_i=200):
        self.model, self.lr, self.betas = model, lr, betas
        self.eps, self.weight_decay, self.t_i = eps, weight_decay, t_i
        self.optimizer = adamw.AMSGradW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.eta = 1.0
        self.now = {'weight_decay': weight_decay, 'lr': lr}

    @staticmethod
    def cos(t_cur, t_i):
        return 0.5 + 0.5 * math.cos(math.pi * t_cur / t_i)

    def __call__(self, i):
        # updat eta every epoch
        self.eta = self.cos(i, self.t_i)
        for p in self.optimizer.param_groups:
            # lr
            p['lr'] = self.eta * self.lr
            # weight decay
            p['weight_decay'] = self.eta * self.weight_decay * math.sqrt(1.0 / (i + 1))
            self.now['weight_decay'], self.now['lr'] = p['weight_decay'], p['lr']
        self.info()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def info(self):
        write('Optimizer: AdamW')
        keys = self.__dict__.keys()
        for key in keys:
            if key == 'model':
                continue
            else:
                write('    {}: {}'.format(key, self.__dict__[key]))


class AdamWLTD(object):

    def __init__(self, model, lr=1.0e-3, betas=(0.9, 0.999), eps=1.0e-8, weight_decay=0.025 * math.sqrt(64 / 50000), t_i=200, ltp_ratio=0.1):
        self.model, self.lr, self.betas = model, lr, betas
        self.eps, self.weight_decay, self.t_i = eps, weight_decay, t_i
        self.ltp_ratio = ltp_ratio
        self.optimizer = adamw.AdamW(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        self.eta = 1.0
        self.now = {'weight_decay': weight_decay, 'lr': lr}

    @staticmethod
    def cos(t_cur, t_i):
        return 0.5 + 0.5 * math.cos(math.pi * t_cur / t_i)

    def __call__(self, i):
        # updat eta every epoch
        self.eta = self.cos(i, self.t_i)
        for p in self.optimizer.param_groups:
            # lr
            p['lr'] = self.eta * self.lr
            # weight decay
            p['weight_decay'] = self.eta * self.weight_decay * math.sqrt(1.0 / (i + 1))
            self.now['weight_decay'], self.now['lr'] = p['weight_decay'], p['lr']
        self.info()

    def step(self):
        for p in self.optimizer.param_groups:
            for var in p['params']:
                weight = var.data
                grad = var.grad.data
                positive_weight_mask = (weight >= 0).type(weight.type())
                negative_weight_mask = (weight < 0).type(weight.type())
                positive_grad_mask = (grad >= 0).type(grad.type())
                negative_grad_mask = (grad < 0).type(grad.type())
                if self.ltp_ratio >= np.random.rand():
                    var.grad.data = grad * (positive_weight_mask * negative_grad_mask + negative_weight_mask * positive_grad_mask)
                else:
                    var.grad.data = grad * (positive_weight_mask * positive_grad_mask + negative_weight_mask * negative_grad_mask)
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def info(self):
        write('Optimizer: AdamW')
        keys = self.__dict__.keys()
        for key in keys:
            if key == 'model':
                continue
            else:
                write('    {}: {}'.format(key, self.__dict__[key]))
