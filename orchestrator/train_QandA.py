import os
os.environ['CUDA_VISIBLE_DEVICES']= '3'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import time
import torch
from torch import optim
from QandA import QandA
from dataset import Slakh_Pop909_Dataset, collate_fn
from torch.utils.data import DataLoader
from scheduler import MinExponentialLR, OptimizerScheduler, TeacherForcingScheduler, ConstantScheduler, ParameterScheduler
from utils import kl_anealing, SummaryWriters, LogPathManager, epoch_time
from tqdm import tqdm


DEVICE = 'cuda:0'
PARALLEL = False
BATCH_SIZE = 128
TRF_LAYERS = 2
N_EPOCH = 30
CLIP = 3
WEIGHTS = [1, 1]
BETA = 1e-2
TFR = [(0.6, 0), (0.5, 0), (0.5, 0)]
LR = 1e-3

MODEL_NAME = 'VQ-Q&A'
DEBUG = 0


model = QandA(name=MODEL_NAME, trf_layers=TRF_LAYERS, device=DEVICE)
if PARALLEL:
    model = torch.nn.DataParallel(model, device_ids=[0, 1])
model.to(DEVICE)


if DEBUG:
    save_root = './save'
    log_path_name = 'debug'
    VERBOSE = True
else:
    save_root = "/data1/AccoMontage3/"
    log_path_name = MODEL_NAME
    VERBOSE = False


slakh_dir = '/data1/Q&A/slakh2100_flac_redux/4_bin_quantization/'
pop909_dir = '/data1/Q&A/POP909-Dataset/quantization/POP09-PIANOROLL-4-bin-quantization/'
train_set = Slakh_Pop909_Dataset(slakh_dir, pop909_dir, hop_len=1, debug_mode=DEBUG, split='train', mode='train')
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, DEVICE))
val_set = Slakh_Pop909_Dataset(slakh_dir, pop909_dir, hop_len=2, debug_mode=DEBUG, split='validation', mode='train')
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, DEVICE, pitch_shift=False))

print(f'Dataset loaded. {len(train_loader)} samples for train and {len(val_loader)} samples for validation.')

optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = MinExponentialLR(optimizer, gamma=0.9999, minimum=1e-5)
#scheduler = None
optimizer_scheduler = OptimizerScheduler(optimizer, scheduler, CLIP)
tfr1_scheduler = TeacherForcingScheduler(*TFR[0], scaler=N_EPOCH*len(train_loader))
tfr2_scheduler = TeacherForcingScheduler(*TFR[1], scaler=N_EPOCH*len(train_loader))
tfr3_scheduler = TeacherForcingScheduler(*TFR[2], scaler=N_EPOCH*len(train_loader))
weights_scheduler = ConstantScheduler(WEIGHTS)
beta_scheduler = TeacherForcingScheduler(BETA, 0, scaler=N_EPOCH*len(train_loader), f=kl_anealing)

params_dic = dict(tfr1=tfr1_scheduler, tfr2=tfr2_scheduler,
                    tfr3=tfr3_scheduler,
                    beta=beta_scheduler, weights=weights_scheduler)
param_scheduler = ParameterScheduler(**params_dic)


readme_fn = 'orchestrator/train_QandA.py'
log_path_mng = LogPathManager(readme_fn, save_root=save_root, log_path_name=log_path_name)


writer_names = ['loss', 'pno_tree_l', 'pl', 'dl', \
                'kl_l', 'kl_sym', 'kl_trf', \
                'feat_l', 'onset_l', 'intensity_l', 'center_l', \
                'func_l', 'fp_l', 'ft_l', 'cmt_p', 'cmt_t', 'plty_p', 'plty_t']

tags = {'loss': None}
loss_writers = SummaryWriters(writer_names, tags, log_path_mng.writer_path)

scheduler_writer_names = ['tfr1', 'tfr2', 'tfr3', 'beta', 'lr']
tags = {'scheduler': None}
scheduler_writers = SummaryWriters(scheduler_writer_names, tags, log_path_mng.writer_path)
deadcode_writer_names = ['fp_usage', 'ft_usage', 'fp_dead', 'ft_dead']
tags = {'deadcode': None}
deadcode_writers = SummaryWriters(deadcode_writer_names, tags, log_path_mng.writer_path)


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
        print(f"\t Pitch loss: {loss['pl']:.3f}")
        print(f"\t Duration loss: {loss['dl']:.3f}")
        print(f"\t Feature loss [onset/intensity/center]: {loss['onset_l']:.3f}/{loss['intensity_l']:.3f}/{loss['center_l']:.3f}")
        print(f"\t KL loss [sym/trf]: {loss['kl_sym']:.3f}/{loss['kl_trf']:.3f}")
        print(f"\t Function loss [pitch/time]: {loss['fp_l']:.6f}/{loss['ft_l']:.6f}")
        print(f"\t Commitment loss [pitch/time]: {loss['cmt_p']:.6f}/{loss['cmt_t']:.6f}")
        print(f"\t Perplexity [pitch/time]: {loss['plty_p']:.6f}/{loss['plty_t']:.6f}")


def scheduler_show(param_scheduler, optimizer_scheduler, verbose=False):
    schedule_params = {}
    schedule_params['tfr1'] = param_scheduler.schedulers['tfr1'].get_tfr()
    schedule_params['tfr2'] = param_scheduler.schedulers['tfr2'].get_tfr()
    schedule_params['tfr3'] = param_scheduler.schedulers['tfr3'].get_tfr()
    schedule_params['beta'] = param_scheduler.schedulers['beta'].get_tfr()
    schedule_params['lr'] = optimizer_scheduler.optimizer.param_groups[0]['lr']
    if verbose:
        print(schedule_params)
    return schedule_params
    

def train(model, dataloader, param_scheduler, device, optimizer_scheduler, writer_names, loss_writers, scheduler_writers, n_epoch):
    model.train()
    param_scheduler.train()
    epoch_loss_dic = init_loss_dic(writer_names)

    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        #try:
        optimizer_scheduler.optimizer_zero_grad()
        
        input_params = param_scheduler.step()
        outputs = model('loss', *batch, **input_params)
        if PARALLEL:
            outputs = tuple([x.mean() for x in outputs])
        loss = outputs[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_scheduler.clip)
        optimizer_scheduler.step()

        epoch_loss_dic = accumulate_loss_dic(writer_names, epoch_loss_dic, outputs)
        batch_loss_dic = batch_loss_dic = write_loss_to_dic(writer_names, outputs)
        train_step = n_epoch * len(dataloader) + idx
        loss_writers.write_task('train', batch_loss_dic, train_step)

        scheduler_dic = scheduler_show(param_scheduler, optimizer_scheduler, verbose=VERBOSE)
        scheduler_writers.write_task('train', scheduler_dic, train_step)

        batch_report(batch_loss_dic, n_epoch, idx, len(dataloader), mode='train', verbose=VERBOSE)

        
        if (idx!=0) and (idx % (len(dataloader) // 10)) == 0:
            batch_z = model.func_time_enc.batch_z[torch.logical_not(batch[4].reshape(-1))]
            batch_z = batch_z.reshape(-1, batch_z.shape[-1])
            ft_usage, ft_deadcode = model.func_time_enc.vq_quantizer.random_restart(batch_z)
            model.func_time_enc.vq_quantizer.reset_usage()
            #if (idx!=0) and (idx % (len(dataloader) // 10)) == 0:
            batch_z = model.func_pitch_enc.batch_z[torch.logical_not(batch[4].reshape(-1))]
            fp_usage, fp_deadcode = model.func_pitch_enc.vq_quantizer.random_restart(batch_z)
            model.func_pitch_enc.vq_quantizer.reset_usage()
            deadcode_writers.write_task('val', dict({'fp_usage': fp_usage, 'ft_usage': ft_usage, 'fp_dead': fp_deadcode, 'ft_dead': ft_deadcode}), train_step)
            if VERBOSE:
                print(f'\t code usage [fp/ft]: {fp_usage}/{ft_usage}', flush=True)
                print(f'\t dead code [fp/ft]: {fp_deadcode}/{ft_deadcode}', flush=True)

            
        #except Exception as exc:
        #    print(exc)
        #    continue

    scheduler_show(param_scheduler, optimizer_scheduler, verbose=True)
    epoch_loss_dic = average_epoch_loss(epoch_loss_dic, len(dataloader))
    

    return epoch_loss_dic


def val(model, dataloader, param_scheduler, device, writer_names, summary_writers, deadcode_writers, n_epoch):
    model.eval()
    param_scheduler.eval()
    epoch_loss_dic = init_loss_dic(writer_names)
    
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        try:
            input_params = param_scheduler.step()
            with torch.no_grad():
                outputs = model('loss', *batch, **input_params)
                if PARALLEL:
                    outputs = tuple([x.mean() for x in outputs])
            #loss = outputs[0]
            
            epoch_loss_dic = accumulate_loss_dic(writer_names, epoch_loss_dic, outputs)
            batch_loss_dic = write_loss_to_dic(writer_names, outputs)
            batch_report(batch_loss_dic, n_epoch, idx, len(dataloader), mode='validation', verbose=VERBOSE)
            #val_step = n_epoch * len(dataloader) + idx
            #summary_writers.write_task('val', batch_loss_dic, val_step)
        except Exception as exc:
            print(exc)
            continue

    epoch_loss_dic = average_epoch_loss(epoch_loss_dic, len(dataloader))
    summary_writers.write_task('val', epoch_loss_dic, n_epoch)

    return epoch_loss_dic


def epoch_report(start_time, end_time, train_loss, valid_loss, n_epoch):
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        print(f'Epoch: {n_epoch + 1:02} | '
              f'Time: {epoch_mins}m {epoch_secs}s',
              flush=True)
        print(f'\tTrain Loss: {train_loss:.3f}', flush=True)
        print(f'\t Valid. Loss: {valid_loss:.3f}', flush=True)


best_valid_loss = float('inf')
for n_epoch in range(N_EPOCH):
    start_time = time.time()
    print(f'Training epoch {n_epoch}')
    train_loss = train(model, train_loader, param_scheduler, DEVICE, optimizer_scheduler, writer_names, loss_writers, scheduler_writers, n_epoch)['loss']
    print(f'Validating epoch {n_epoch}')
    val_loss = val(model, val_loader, param_scheduler, DEVICE, writer_names, loss_writers, deadcode_writers, n_epoch)['loss']
    end_time = time.time()

    if PARALLEL:
        torch.save(model.module.state_dict(), log_path_mng.epoch_model_path(f'{MODEL_NAME}_{str(n_epoch).zfill(3)}'))
    else:
        torch.save(model.state_dict(), log_path_mng.epoch_model_path(f'{MODEL_NAME}_{str(n_epoch).zfill(3)}'))

    if val_loss < best_valid_loss:
        best_valid_loss = val_loss
        if PARALLEL:
            torch.save(model.module.state_dict(), log_path_mng.valid_model_path(MODEL_NAME))
        else:
            torch.save(model.state_dict(), log_path_mng.valid_model_path(MODEL_NAME))
    
    epoch_report(start_time, end_time, train_loss, val_loss, n_epoch)
if PARALLEL:
    torch.save(model.module.state_dict(), log_path_mng.final_model_path(MODEL_NAME))
else:
    torch.save(model.state_dict(), log_path_mng.final_model_path(MODEL_NAME))


