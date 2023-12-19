from .training import scheduled_sampling
from torch.optim.lr_scheduler import ExponentialLR


class MinExponentialLR(ExponentialLR):
    def __init__(self, optimizer, gamma, minimum, last_epoch=-1):
        self.min = minimum
        super(MinExponentialLR, self).__init__(optimizer, gamma, last_epoch=-1)

    def get_lr(self):
        return [
            max(base_lr * self.gamma ** self.last_epoch, self.min)
            for base_lr in self.base_lrs
        ]


class _Scheduler:

    def __init__(self, step=0, mode='train'):
        self._step = step
        self._mode = mode

    def _update_step(self):
        if self._mode == 'train':
            self._step += 1
        elif self._mode == 'val':
            pass
        else:
            raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def train(self):
        self._mode = 'train'

    def eval(self):
        self._mode = 'val'


class ConstantScheduler(_Scheduler):

    def __init__(self, param, step=0.):
        super(ConstantScheduler, self).__init__(step)
        self.param = param

    def step(self, scaler=None):
        self._update_step()
        return self.param


class TeacherForcingScheduler(_Scheduler):

    def __init__(self, high, low, scaler, f=scheduled_sampling, step=0):
        super(TeacherForcingScheduler, self).__init__(step)
        self.high = high
        self.low = low
        self._step = step
        self.scaler = scaler
        self.schedule_f = f

    def get_tfr(self):
        return self.schedule_f(self._step/self.scaler, self.high, self.low)

    def step(self):
        tfr = self.get_tfr()
        self._update_step()
        return tfr


class OptimizerScheduler(_Scheduler):

    def __init__(self, optimizer, scheduler, clip, step=0):
        # optimizer and scheduler are pytorch class
        super(OptimizerScheduler, self).__init__(step)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip = clip

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, require_zero_grad=False):
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        if require_zero_grad:
            self.optimizer_zero_grad()
        self._update_step()


class OptimizerSchedulerWithWarmUp(_Scheduler):

    def __init__(self, optimizer, warmupscheduler, scheduler, clip, warmup_step=1000, step=0):
        # optimizer and scheduler are pytorch class
        super(OptimizerSchedulerWithWarmUp, self).__init__(step)
        self.optimizer = optimizer
        self.warmupscheduler = warmupscheduler
        self.scheduler = scheduler
        self.warmup_step = warmup_step
        self.clip = clip

    def optimizer_zero_grad(self):
        self.optimizer.zero_grad()

    def step(self, require_zero_grad=False):
        self.optimizer.step()
        if self.scheduler is not None:
            if self._step < self.warmup_step:
                self.warmupscheduler.step()
            else:
                self.scheduler.step()
        if require_zero_grad:
            self.optimizer_zero_grad()
        self._update_step()


class ParameterScheduler(_Scheduler):

    def __init__(self, step=0, mode='train', **schedulers):
        # optimizer and scheduler are pytorch class
        super(ParameterScheduler, self).__init__(step)
        self.schedulers = schedulers
        self.mode = mode

    def train(self):
        self.mode = 'train'
        for scheduler in self.schedulers.values():
            scheduler.train()

    def eval(self):
        self.mode = 'val'
        for scheduler in self.schedulers.values():
            scheduler.eval()

    def step(self):
        params_dic = {}
        for key, scheduler in self.schedulers.items():
            params_dic[key] = scheduler.step()
        return params_dic





