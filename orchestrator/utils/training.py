import os
import datetime
import shutil
import torch
import numpy as np
from torch.distributions import Normal, kl_divergence
from torch.utils.tensorboard import SummaryWriter


def get_zs_from_dists(dists, sample=False):
    return [dist.rsample() if sample else dist.mean for dist in dists]


def scheduled_sampling(i, high=0.7, low=0.05, scaler=1e5):
    x = 10 * (i - 0.5)
    z = 1 / (1 + np.exp(x))
    y = (high - low) * z + low
    return y


def kl_anealing(i, high=0.1, low=0., scaler=None):
    hh = 1 - low
    ll = 1 - high
    x = 10 * (i - 0.5)
    z = 1 / (1 + np.exp(x))
    y = (hh - ll) * z + ll
    return 1 - y


def standard_normal(shape, device):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    #if torch.cuda.is_available():
    N.loc = N.loc.to(device)
    N.scale = N.scale.to(device)
    return N


def kl_with_normal(dist):
    shape = dist.mean.size(-1)
    normal = standard_normal(shape, dist.mean.device)
    kl = kl_divergence(dist, normal).mean()
    return kl


class SummaryWriters:

    def __init__(self, writer_names, tags, log_path, tasks=('train', 'val')):
        # writer_names example: ['loss', 'kl_loss', 'recon_loss']
        # tags example: {'name1': None, 'name2': (0, 1)}
        self.log_path = log_path
        #assert 'loss' == writer_names[0]
        self.writer_names = writer_names
        self.tags = tags
        self._regularize_tags()

        writer_dic = {}
        for name in writer_names:
            writer_dic[name] = SummaryWriter(os.path.join(log_path, name))
        self.writers = writer_dic

        all_tags = {}
        for task in tasks:
            task_dic = {}
            for key, val in self.tags.items():
                task_dic['_'.join([task, key])] = val
            all_tags[task] = task_dic
        self.all_tags = all_tags

    def _init_summary_writer(self):
        tags = {'batch_train': (0, 1, 2, 3, 4)}
        self.summary_writers = SummaryWriters(self.writer_names, tags,
                                              self.writer_path)

    def _regularize_tags(self):
        for key, val in self.tags.items():
            if val is None:
                self.tags[key] = tuple(range(len(self.writer_names)))

    def single_write(self, name, tag, val, step):
        self.writers[name].add_scalar(tag, val, step)

    def write_tag(self, task, tag, vals, step):
        assert len(vals) == len(self.all_tags[task][tag])
        for name_id, val in zip(self.all_tags[task][tag], vals):
            name = self.writer_names[name_id]
            self.single_write(name, tag, val, step)

    def write_task(self, task, vals_dic, step):
        for tag, name_ids in self.all_tags[task].items():
            vals = [vals_dic[self.writer_names[i]] for i in name_ids]
            self.write_tag(task, tag, vals, step)


def join_fn(*items, ext='pt'):
    return '.'.join(['_'.join(items), ext])


class LogPathManager:

    def __init__(self, readme_fn=None, save_root='.', log_path_name='result',
                 with_date=True, with_time=True,
                 writer_folder='writers', model_folder='models'):
        date = str(datetime.date.today()) if with_date else ''
        ctime = datetime.datetime.now().time().strftime("%H%M%S") \
            if with_time else ''
        log_folder = '_'.join([date, ctime, log_path_name])
        log_path = os.path.join(save_root, log_folder)
        writer_path = os.path.join(log_path, writer_folder)
        model_path = os.path.join(log_path, model_folder)
        self.log_path = log_path
        self.writer_path = writer_path
        self.model_path = model_path
        LogPathManager.create_path(log_path)
        LogPathManager.create_path(writer_path)
        LogPathManager.create_path(model_path)
        if readme_fn is not None:
            shutil.copyfile(readme_fn, os.path.join(log_path, 'readme.txt'))

    @staticmethod
    def create_path(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def epoch_model_path(self, model_name):
        model_fn = join_fn(model_name, 'epoch', ext='pt')
        return os.path.join(self.model_path, model_fn)

    def valid_model_path(self, model_name):
        model_fn = join_fn(model_name, 'valid', ext='pt')
        return os.path.join(self.model_path, model_fn)

    def final_model_path(self, model_name):
        model_fn = join_fn(model_name, 'final', ext='pt')
        return os.path.join(self.model_path, model_fn)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
