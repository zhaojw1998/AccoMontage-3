import os
import time
import torch
from torch import optim
from Prior import Prior
from vq_dataset import VQ_LMD_Dataset, collate_fn
from torch.utils.data import DataLoader
from scheduler import MinExponentialLR, OptimizerScheduler, TeacherForcingScheduler, ConstantScheduler, ParameterScheduler
from utils import SummaryWriters, LogPathManager, epoch_time
from tqdm import tqdm

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


def main(rank, world_size, log_path_mng, VERBOSE, MODEL_NAME):
    #print('rank:', rank)
    ddp_setup(rank, world_size)

    PRETRAIN_PATH = "data_file_dir/params_qa.pt"
    BATCH_SIZE = 32
    N_EPOCH = 30
    CLIP = 3
    LR = 1e-3

    if VERBOSE:
        N_EPOCH=10

    model = Prior.init_model(pretrain_model_path=PRETRAIN_PATH, DEVICE=rank)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)   

    lmd_dir = "/data1/LMD/vector_quantization_029/"
    train_set = VQ_LMD_Dataset(lmd_dir, debug_mode=VERBOSE, split='train', mode='train')
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, rank), sampler=DistributedSampler(train_set))
    val_set = VQ_LMD_Dataset(lmd_dir, debug_mode=VERBOSE, split='validation', mode='train')
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, rank), sampler=DistributedSampler(val_set))
    print(f'Dataset loaded. {len(train_loader)} samples for train and {len(val_loader)} samples for validation.')


    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = MinExponentialLR(optimizer, gamma=0.99996, minimum=1e-5)
    #scheduler = None
    optimizer_scheduler = OptimizerScheduler(optimizer, scheduler, CLIP)
    #tfr_scheduler = TeacherForcingScheduler(*TFR, scaler=N_EPOCH*len(train_loader))
    #params_dic = dict(tfr=tfr_scheduler)
    param_scheduler = None #ParameterScheduler(**params_dic)

    writer_names = ['loss', 'fp_l', 'ft_l']
    scheduler_writer_names = ['lr']
    
    if rank == 0:
        tags = {'loss': None}
        loss_writers = SummaryWriters(writer_names, tags, log_path_mng.writer_path)
        tags = {'scheduler': None}
        scheduler_writers = SummaryWriters(scheduler_writer_names, tags, log_path_mng.writer_path)
    else:
        loss_writers = None
        scheduler_writers = None


    #best_valid_loss = float('inf')
    for n_epoch in range(N_EPOCH):
        start_time = time.time()
        train_loader.sampler.set_epoch(n_epoch)
        print(f'Training epoch {n_epoch}')
        train_loss = train(model, train_loader, param_scheduler, optimizer_scheduler, writer_names, loss_writers, scheduler_writers, n_epoch=n_epoch, VERBOSE=VERBOSE)['loss']
        print(f'Validating epoch {n_epoch}')
        val_loss = val(model, val_loader, param_scheduler, writer_names, loss_writers, n_epoch=n_epoch, VERBOSE=VERBOSE)['loss']
        end_time = time.time()

        if rank == 0:
            torch.save(model.module.state_dict(), log_path_mng.epoch_model_path(f'{MODEL_NAME}_{str(n_epoch).zfill(3)}'))

        #if val_loss < best_valid_loss:
        #    best_valid_loss = val_loss
        #    if rank == 0:
        #        torch.save(model.module.state_dict(), log_path_mng.valid_model_path(MODEL_NAME))
        
        epoch_report(start_time, end_time, train_loss, val_loss, n_epoch)
        #if rank == 0:
        #    torch.save(model.module.state_dict(), log_path_mng.final_model_path(MODEL_NAME))

    destroy_process_group()



def accumulate_loss_dic(writer_names, loss_dic, loss_items):
        assert len(writer_names) == len(loss_items)
        for key, val in zip(writer_names, loss_items):
            loss_dic[key] += val.item()
        return loss_dic

def write_loss_to_dic(writer_names, loss_items):
    loss_dic = {}
    assert len(writer_names) == len(loss_items)
    for key, val in zip(writer_names, loss_items):
        loss_dic[key] = val.item()
    return loss_dic

def init_loss_dic(writer_names):
        loss_dic = {}
        for key in writer_names:
            loss_dic[key] = 0.
        return loss_dic

def average_epoch_loss(epoch_loss_dict, num_batch):
    for key in epoch_loss_dict:
            epoch_loss_dict[key] /= num_batch
    return epoch_loss_dict


def batch_report(loss, n_epoch, idx, num_batch, mode='training', verbose=False):
    if verbose:
        print(f'------------{mode}------------')
        print('Epoch: [{0}][{1}/{2}]'.format(n_epoch, idx, num_batch))
        print(f"\t Total loss: {loss['loss']}")
        print(f"\t pitch func loss: {loss['fp_l']:.3f}")
        print(f"\t time func loss: {loss['ft_l']:.3f}")


def scheduler_show(param_scheduler, optimizer_scheduler, verbose=False):
    schedule_params = {}
    #schedule_params['tfr'] = param_scheduler.schedulers['tfr'].get_tfr()
    schedule_params['lr'] = optimizer_scheduler.optimizer.param_groups[0]['lr']
    if verbose:
        print(schedule_params)
    return schedule_params
    

def train(model, dataloader, param_scheduler, optimizer_scheduler, writer_names, loss_writers, scheduler_writers, n_epoch, VERBOSE):
    model.train()
    #param_scheduler.train()
    epoch_loss_dic = init_loss_dic(writer_names)

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        try:
            optimizer_scheduler.optimizer_zero_grad()
            
            #input_params = param_scheduler.step()
            outputs = model('loss', *batch)#, **input_params)
            loss = outputs[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_scheduler.clip)
            optimizer_scheduler.step()

            epoch_loss_dic = accumulate_loss_dic(writer_names, epoch_loss_dic, outputs)
            batch_loss_dic = write_loss_to_dic(writer_names, outputs)
            train_step = n_epoch * len(dataloader) + idx
            if loss_writers is not None:
                loss_writers.write_task('train', batch_loss_dic, train_step)
                batch_report(batch_loss_dic, n_epoch, idx, len(dataloader), mode='train', verbose=VERBOSE)

            scheduler_dic = scheduler_show(param_scheduler, optimizer_scheduler, verbose=VERBOSE)
            if scheduler_writers is not None:
                scheduler_writers.write_task('train', scheduler_dic, train_step)
        except Exception as exc:
            print(exc)
            print(batch[0].shape, batch[1].shape)
            continue

    scheduler_show(param_scheduler, optimizer_scheduler, verbose=True)
    epoch_loss_dic = average_epoch_loss(epoch_loss_dic, len(dataloader))
    return epoch_loss_dic


def val(model, dataloader, param_scheduler, writer_names, summary_writers, n_epoch, VERBOSE):
    model.eval()
    #param_scheduler.eval()
    epoch_loss_dic = init_loss_dic(writer_names)
    
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        try:
            #input_params = param_scheduler.step()
            with torch.no_grad():
                outputs = model('loss', *batch)#, **input_params)
            epoch_loss_dic = accumulate_loss_dic(writer_names, epoch_loss_dic, outputs)
            batch_loss_dic = write_loss_to_dic(writer_names, outputs)
            if summary_writers is not None:
                batch_report(batch_loss_dic, n_epoch, idx, len(dataloader), mode='validation', verbose=VERBOSE)
            #val_step = n_epoch * len(dataloader) + idx
            #summary_writers.write_task('val', batch_loss_dic, val_step)
        except Exception as exc:
            print(exc)
            print(batch[0].shape, batch[1].shape)
            continue
    epoch_loss_dic = average_epoch_loss(epoch_loss_dic, len(dataloader))
    if summary_writers is not None:
        summary_writers.write_task('val', epoch_loss_dic, n_epoch)
    return epoch_loss_dic


def epoch_report(start_time, end_time, train_loss, valid_loss, n_epoch):
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {n_epoch + 1:02} | '
              f'Time: {epoch_mins}m {epoch_secs}s',
              flush=True)
        print(f'\tTrain Loss: {train_loss:.3f}', flush=True)
        print(f'\t Valid. Loss: {valid_loss:.3f}', flush=True)





if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']= '2, 3'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    MODEL_NAME = 'Prior Model'
    DEBUG = 0

    if DEBUG:
        save_root = './save'
        log_path_name = 'debug'
    else:
        save_root = '/data1/AccoMontage3/'
        log_path_name = MODEL_NAME


    readme_fn = 'orchestrator/train_Prior_DDP.py'
    log_path_mng = LogPathManager(readme_fn, save_root=save_root, log_path_name=log_path_name)

    world_size = torch.cuda.device_count()
    #print(world_size)
    mp.spawn(main, args=(world_size, log_path_mng, DEBUG, MODEL_NAME), nprocs=world_size)
