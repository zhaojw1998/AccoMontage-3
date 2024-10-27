import os
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import random


ACC = 16            #quantize every 4 beats into 16 positions
SAMPLE_LEN = 32     #each sample has 2 bars
BAR_HOP_LEN = 1     #hop size is 1 bar
AUG_P = np.array([2, 2, 5, 5, 3, 7, 7, 5, 7, 3, 5, 1])  #prior for pitch transposition augmentation
NUM_INSTR_CLASS = 34    #number of supported instruments

#Supported instrument programs in Slakh2100 dataset
SLAKH_CLASS_PROGRAMS = dict({
    0: 'Acoustic Piano',    #0
    4: 'Electric Piano',    #1
    8: 'Chromatic Percussion',#2
    16: 'Organ',    #3
    24: 'Acoustic Guitar',  #4
    26: 'Clean Electric Guitar',    #5
    29: 'Distorted Electric Guitar',    #6
    32: 'Acoustic Bass',    #7
    33: 'Electric Bass',    #8
    40: 'Violin',   #9
    41: 'Viola',    #10
    42: 'Cello',    #11
    43: 'Contrabass',   #12
    46: 'Orchestral Harp',  #13
    47: 'Timpani',  #14
    48: 'String Ensemble',  #15
    50: 'Synth Strings',    #16
    52: 'Choir and Voice',  #17
    55: 'Orchestral Hit',   #18
    56: 'Trumpet',  #19
    57: 'Trombone', #20
    58: 'Tuba', #21
    60: 'French Horn',  #22
    61: 'Brass Section',    #23
    64: 'Soprano/Alto Sax', #24
    66: 'Tenor Sax',    #25
    67: 'Baritone Sax', #26
    68: 'Oboe', #27
    69: 'English Horn', #28
    70: 'Bassoon',  #29
    71: 'Clarinet', #30
    72: 'Pipe', #31
    80: 'Synth Lead',   #32
    88: 'Synth Pad' #33
})

#map an arbituary program to a supported Slakh2100 program
#re-calibrated based on https://github.com/ethman/slakh-generation/blob/master/instr_defs_metadata/komplete_strict.json
SLAKH_PROGRAM_MAPPING = dict({0: 0, 1: 0, 2: 4, 3: 0, 4: 4, 5: 4, 6: 0, 7: 4,\
                8: 8, 9: 8, 10: 8, 11: 8, 12: 8, 13: 8, 14: 8, 15: 8,\
                16: 16, 17: 16, 18: 16, 19: 16, 20: 16, 21: 16, 22: 16, 23: 16,\
                24: 24, 25: 24, 26: 26, 27: 26, 28: 26, 29: 29, 30: 29, 31: 26,\
                32: 32, 33: 33, 34: 33, 35: 33, 36: 33, 37: 33, 38: 33, 39: 33,\
                40: 40, 41: 41, 42: 42, 43: 43, 44: 48, 45: 48, 46: 46, 47: 47,\
                48: 48, 49: 48, 50: 50, 51: 50, 52: 52, 53: 52, 54: 52, 55: 55,\
                56: 56, 57: 57, 58: 58, 59: 56, 60: 60, 61: 61, 62: 61, 63: 61,\
                64: 64, 65: 64, 66: 66, 67: 67, 68: 68, 69: 69, 70: 70, 71: 71,\
                72: 72, 73: 72, 74: 72, 75: 72, 76: 72, 77: 72, 78: 72, 79: 72,\
                80: 80, 81: 80, 82: 80, 83: 80, 84: 80, 85: 80, 86: 80, 87: 80,\
                88: 88, 89: 88, 90: 88, 91: 88, 92: 88, 93: 88, 94: 88, 95: 88})

#map Slakh2100 programs to sequential indices for embedding purposes
EMBED_PROGRAM_MAPPING = dict({
    0: 0, 4: 1, 8: 2, 16: 3, 24: 4, 26: 5, 29: 6, 32: 7,\
    33: 8, 40: 9, 41: 10, 42: 11, 43: 12, 46: 13, 47: 14, 48: 15,\
    50: 16, 52: 17, 55: 18, 56: 19, 57: 20, 58: 21, 60: 22, 61: 23, 
    64: 24, 66: 25, 67: 26, 68: 27, 69: 28, 70: 29, 71: 30, 72: 31,\
    80: 32, 88: 33})


class Slakh2100_Pop909_Dataset(Dataset):
    def __init__(self, slakh_dir, pop909_dir, sample_len=SAMPLE_LEN, hop_len=BAR_HOP_LEN, debug_mode=False, split='train', mode='train', with_dynamics=False, merge_pop909=0):
        super(Slakh2100_Pop909_Dataset, self).__init__()
        self.split = split
        self.mode = mode
        self.debug_mode = debug_mode

        self.with_dynamics = with_dynamics
        self.merge_pop909 = merge_pop909

        self.memory = dict({'tracks': [],
                            'programs': [],
                            'dynamics': [],
                            'dir': []
                            })
        self.anchor_list = []
        self.sample_len = sample_len
        
        if slakh_dir is not None:
            print('loading Slakh2100 Dataset ...')
            self.load_data(slakh_dir, sample_len, hop_len)
        if pop909_dir is not None:
            print('loading Pop909 Dataset ...')
            self.load_data(pop909_dir, sample_len, hop_len)

    def __len__(self):
        return len(self.anchor_list)
    
    def __getitem__(self, idx):
        song_id, start = self.anchor_list[idx]

        if self.mode == 'train': 
            tracks_sample = self.memory['tracks'][song_id][:, start: start+self.sample_len]
            program_sample = self.memory['programs'][song_id]
            #delete empty tracks if any
            #non_empty = np.nonzero(np.sum(tracks_sample, axis=(1, 2)))[0]
            #if len(non_empty) > 0:
            #    tracks_sample = tracks_sample[non_empty]
            #    program_sample = program_sample[non_empty]

        elif (self.mode == 'test') or (self.mode == 'inference'): 
            tracks_sample = self.memory['tracks'][song_id][:, start:]
            program_sample = self.memory['programs'][song_id]

        if ((len(program_sample) <= 3) and (program_sample == 0).all()):
            #merge pop909 into a single piano track at certain probability
            if np.random.rand() < self.merge_pop909:    
                tracks_sample = np.max(tracks_sample, axis=0, keepdims=True)
                program_sample = np.array([0])

        if self.with_dynamics:
            dynamics = self.memory['dynamics'][song_id][:, start: start+self.sample_len]
        else: 
            dynamics = None
        
        return tracks_sample, program_sample, dynamics, self.memory['dir'][song_id]


    def slakh_program_mapping(self, programs):
        return np.array([EMBED_PROGRAM_MAPPING[SLAKH_PROGRAM_MAPPING[program]] for program in programs])


    def load_data(self, data_dir, sample_len, hop_len):
        if self.split == 'inference':
            song_list = [os.path.join(data_dir, 'validation', item) for item in os.listdir(os.path.join(data_dir, 'validation'))] \
                + [os.path.join(data_dir, 'test', item) for item in os.listdir(os.path.join(data_dir, 'test'))]
        else:
            song_list = [os.path.join(data_dir, self.split, item) for item in os.listdir(os.path.join(data_dir, self.split))]
        if self.debug_mode:
            song_list = song_list[: 10]
        for song_dir in tqdm(song_list):
            song_data = np.load(song_dir)
            tracks = song_data['tracks']   #(n_track, time, 128)
            if 'programs' in song_data:
                programs = song_data['programs']    #(n_track, )
            else:
                programs = np.array([0]*len(tracks))

            center_pitch = compute_center_pitch(tracks)
            pitch_sort = np.argsort(center_pitch)[::-1]
            tracks = tracks[pitch_sort]
            programs = programs[pitch_sort]

            """clipping""" 
            if self.mode == 'train':
                if self.split =='validation':
                    # during model training, no overlapping for validation set
                    for i in range(0, tracks.shape[1], sample_len):
                        if i + sample_len >= tracks.shape[1]:
                            break
                        self.anchor_list.append((len(self.memory['tracks']), i))  #(song_id, start, total_length)
                else:
                    # otherwise, hop size is 1-bar
                    downbeats = np.nonzero(song_data['db_indicator'])[0]
                    for i in range(0, len(downbeats), hop_len):
                        if downbeats[i] + sample_len >= tracks.shape[1]:
                            break
                        self.anchor_list.append((len(self.memory['tracks']), downbeats[i]))  #(song_id, start)

            elif (self.mode == 'test') or (self.mode == 'inference'):
                start = np.nonzero(song_data['db_indicator'])[0][0]
                end = start + (tracks.shape[1] - start) // sample_len * sample_len
                if end < tracks.shape[1]:
                    pad_len = end + sample_len - tracks.shape[1]
                    end += sample_len
                    tracks = np.pad(tracks, ((0, 0), (0, pad_len), (0, 0)), mode='constant', constant_values=(0,))
                tracks = tracks[:, start: end]
                self.anchor_list.append((len(self.memory['tracks']), start))

            self.memory['tracks'].append(tracks)
            self.memory['programs'].append(self.slakh_program_mapping(programs))
            self.memory['dir'].append(song_dir)

            if self.with_dynamics:
                self.memory['dynamics'].append(song_data['dynamics'])


def collate_fn(batch, device, pitch_shift=True):
    #print(batch)
    max_tracks = max([max(len(item[0]), 1) for item in batch])

    tracks = [] 
    pno_reduction = []
    instrument = []
    aux_feature = []
    mask = []   #track-wise pad mask
    function = []

    if pitch_shift:
        aug_p = AUG_P / AUG_P.sum()
        aug_shift = np.random.choice(np.arange(-6, 6), 1, p=aug_p)[0]
    else:
        aug_shift = 0

    for pr, programs, _, _ in batch:
        pr = pr_mat_pitch_shift(pr, aug_shift)
        aux, _, func = compute_pr_feat(pr)
        mask.append([0]*len(pr) + [1]*(max_tracks-len(pr)))

        pr = np.pad(pr, ((0, max_tracks-len(pr)), (0, 0), (0, 0)), mode='constant', constant_values=(0,))
        programs = np.pad(programs, (0, max_tracks-len(programs)), mode='constant', constant_values=(NUM_INSTR_CLASS,))
        aux = np.pad(aux, ((0, max_tracks-len(aux)), (0, 0), (0, 0)), mode='constant', constant_values=(0,))
        func = np.pad(func, ((0, max_tracks-len(func)), (0, 0)), mode='constant', constant_values=(0,))

        mix = pr2grid(np.max(pr, axis=0), max_note_count=32)
        grid = np.array([pr2grid(matrix) for matrix in pr])

        tracks.append(grid)
        pno_reduction.append(mix)
        instrument.append(programs)
        aux_feature.append(aux)
        function.append(func)

    return  torch.from_numpy(np.array(pno_reduction)).long().to(device), \
            torch.from_numpy(np.array(instrument)).to(device), \
            torch.from_numpy(np.array(function)).float().to(device),\
            torch.from_numpy(np.array(tracks)).long().to(device), \
            torch.from_numpy(np.array(aux_feature)).float().to(device), \
            torch.BoolTensor(mask).to(device)


def collate_fn_inference(batch, device):
    assert len(batch) == 1
    tracks, instrument, dynamics, song_dir = batch[0]

    track, time, _ = tracks.shape
    if time % 32 != 0:
        pad_len = (time//32+1)*32 - time
        tracks = np.pad(tracks, ((0, 0), (0, pad_len), (0, 0)))
        if dynamics is not None:
            dynamics = np.pad(dynamics, ((0, 0), (0, pad_len), (0, 0), (0, 0)))
            dynamics[:, -pad_len:, :, -1] = -1
    tracks = tracks.reshape(track, -1, 32, 128).transpose(1, 0, 2, 3)

    _, _, function = compute_pr_feat(tracks)

    pno_reduction = np.array([pr2grid(matrix, max_note_count=32) for matrix in np.max(tracks, axis=1)])

    pno_reduction = torch.from_numpy(pno_reduction).long().to(device)
    instrument = torch.from_numpy(instrument).repeat(tracks.shape[0], 1).to(device)
    function = torch.from_numpy(np.array(function)).float().to(device)

    return (pno_reduction, instrument, function), dynamics, song_dir
    
        
def pr_mat_pitch_shift(pr_mat, shift):
    pr_mat = pr_mat.copy()
    pr_mat = np.roll(pr_mat, shift, -1)
    return pr_mat


def pr2grid(pr_mat, max_note_count=16, max_pitch=127, min_pitch=0,
                       pitch_pad_ind=130, dur_pad_ind=2,
                       pitch_sos_ind=128, pitch_eos_ind=129):
    """pr_mat: (32, 128)"""
    """Z. Wang, et al. Learning Interpretable Representation for Controllable Polyphonic Music Generation, in ISMIR 2020"""
    sample_len = len(pr_mat)
    grid = np.ones((sample_len, max_note_count, 6), dtype=int) * dur_pad_ind
    grid[:, :, 0] = pitch_pad_ind
    grid[:, 0, 0] = pitch_sos_ind
    cur_idx = np.ones(sample_len, dtype=int)
    for t, p in zip(*np.where(pr_mat != 0)):
        if cur_idx[t] == max_note_count - 1:
            continue
        grid[t, cur_idx[t], 0] = p - min_pitch
        binary = np.binary_repr(min(int(pr_mat[t, p]), 32) - 1, width=5)
        grid[t, cur_idx[t], 1: 6] = \
            np.fromstring(' '.join(list(binary)), dtype=int, sep=' ')
        cur_idx[t] += 1
    grid[np.arange(0, sample_len), cur_idx, 0] = pitch_eos_ind
    return grid


def compute_pr_feat(pr):
    #pr: (track, time, 128)
    onset = (np.sum(pr, axis=-1) > 0) * 1.   #(track, time)
    rhy_intensity = np.clip(np.sum((pr > 0) * 1., axis=-1) / 14, a_min=None, a_max=1)    #(track, time)

    weight = np.sum(pr, axis=-1)
    weight[weight==0] = 1
    pitch_center = np.sum(np.arange(0, 128)[np.newaxis, np.newaxis, :] * pr, axis=-1) / weight / 128

    feature = np.stack((onset, rhy_intensity, pitch_center), axis=-1)

    func_pitch = np.sum((pr > 0) * 1., axis=-2) / 32

    func_time = rhy_intensity.copy()
    
    return feature, func_pitch, func_time


def compute_center_pitch(pr):
    #pr: (track, time, 128)
    #pr[pr > 0] = 1
    weight = np.sum(pr, axis=(-2, -1))
    weight[weight == 0] = 1
    pitch_center = np.sum(np.arange(0, 128)[np.newaxis, np.newaxis, :] * pr, axis=(-2, -1)) / weight 
    return pitch_center #(track, )


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    DEVICE = 'cuda:0'
    DEBUG = True
    BATCH_SIZE = 8
    MODE = 'train'
    slakh_dir = "/data1/zhaojw/data_file_dir/Slakh2100_inference_set"
    
    val_set = Slakh2100_Pop909_Dataset(slakh_dir, None, debug_mode=DEBUG, split='validation', mode='train')
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, DEVICE))
    print(f'Dataset loaded. {len(val_loader)} samples for training.')
    for idx, (pno_red, instrument, function, tracks, aux_feature, mask) in tqdm(enumerate(val_loader), total=len(val_loader)):
        print(pno_red.shape)
        print(instrument.shape)
        print(function.shape)
        print(tracks.shape)
        print(aux_feature.shape)
        print(mask.shape)
        break

    