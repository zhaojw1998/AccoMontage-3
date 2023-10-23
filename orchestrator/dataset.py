import os
import numpy as np
import pretty_midi as pyd
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import pandas as pd
from .utils import retrieve_control


ACC = 16
SAMPLE_LEN = 2 * 16
BAR_HOP_LEN = 1
AUG_P = np.array([2, 2, 5, 5, 3, 7, 7, 5, 7, 3, 5, 1])
NUM_INSTR_CLASS = 34
NUM_PITCH_CODE = 64
NUM_TIME_CODE = 128
TOTAL_LEN_BIN = np.array([4, 7, 12, 15, 20, 23, 28, 31, 36, 39, 44, 47, 52, 55, 60, 63, 68, 71, 76, 79, 84, 87, 92, 95, 100, 103, 108, 111, 116, 119, 124, 127, 132])
ABS_POS_BIN = np.arange(129)
REL_POS_BIN = np.arange(128)

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

SLAKH_PROGRAM_MAPPING = dict({0: 0, 1: 0, 2: 0, 3: 0, 4: 4, 5: 4, 6: 4, 7: 4,\
                8: 8, 9: 8, 10: 8, 11: 8, 12: 8, 13: 8, 14: 8, 15: 8,\
                16: 16, 17: 16, 18: 16, 19: 16, 20: 16, 21: 16, 22: 16, 23: 16,\
                24: 24, 25: 24, 26: 26, 27: 26, 28: 26, 29: 29, 30: 29, 31: 29,\
                32: 32, 33: 33, 34: 33, 35: 33, 36: 33, 37: 33, 38: 33, 39: 33,\
                40: 40, 41: 41, 42: 42, 43: 43, 44: 43, 45: 43, 46: 46, 47: 47,\
                48: 48, 49: 48, 50: 50, 51: 50, 52: 52, 53: 52, 54: 52, 55: 55,\
                56: 56, 57: 57, 58: 58, 59: 58, 60: 60, 61: 61, 62: 61, 63: 61,\
                64: 64, 65: 64, 66: 66, 67: 67, 68: 68, 69: 69, 70: 70, 71: 71,\
                72: 72, 73: 72, 74: 72, 75: 72, 76: 72, 77: 72, 78: 72, 79: 72,\
                80: 80, 81: 80, 82: 80, 83: 80, 84: 80, 85: 80, 86: 80, 87: 80,\
                88: 88, 89: 88, 90: 88, 91: 88, 92: 88, 93: 88, 94: 88, 95: 88})

EMBED_PROGRAM_MAPPING = dict({
    0: 0, 4: 1, 8: 2, 16: 3, 24: 4, 26: 5, 29: 6, 32: 7,\
    33: 8, 40: 9, 41: 10, 42: 11, 43: 12, 46: 13, 47: 14, 48: 15,\
    50: 16, 52: 17, 55: 18, 56: 19, 57: 20, 58: 21, 60: 22, 61: 23, 
    64: 24, 66: 25, 67: 26, 68: 27, 69: 28, 70: 29, 71: 30, 72: 31,\
    80: 32, 88: 33})


class Slakh_Dataset(Dataset):
    def __init__(self, slakh_dir, debug_mode=False, split='train', mode='train'):
        super(Slakh_Dataset, self).__init__()
        self.slakh_dir = slakh_dir
        self.split = split
        self.mode = mode
        self.debug_mode = debug_mode
        self.pr_list = []
        self.program_list = [] 
        self.anchor_list = []
        
        self.load_slakh()

    def __len__(self):
        return len(self.anchor_list)
    
    def __getitem__(self, idx):
        song_id, start, total_len = self.anchor_list[idx]
        pr = self.pr_list[song_id][:, start: min(total_len, start+SAMPLE_LEN)]
        prog = self.program_list[song_id]
        return pr, prog, (start, total_len)


    def slakh_program_mapping(self, programs):
        return np.array([SLAKH_PROGRAM_MAPPING[program] for program in programs])


    def load_slakh(self):
        if self.split == 'inference':
            slakh_list = []
            slakh_list += os.listdir(os.path.join(self.slakh_dir, 'validation'))
            slakh_list += os.listdir(os.path.join(self.slakh_dir, 'test'))
        else:
            slakh_list = os.listdir(os.path.join(self.slakh_dir, self.split))
        if self.debug_mode:
            slakh_list = slakh_list[: 10]
        for song in slakh_list:
            if self.split == 'inference':
                if song in os.listdir(os.path.join(self.slakh_dir, 'validation')):
                    slakh_data = np.load(os.path.join(self.slakh_dir, 'validation', song))
                elif song in os.listdir(os.path.join(self.slakh_dir, 'test')):
                    slakh_data = np.load(os.path.join(self.slakh_dir, 'test', song))
            else:
                slakh_data = np.load(os.path.join(self.slakh_dir, self.split, song))
            tracks = slakh_data['tracks']   #(n_track, time, 128)
            programs = slakh_data['programs']    #(n_track, )
            db_indicator = slakh_data['db_indicator']  #(time, )
            """padding"""
            num_bars = int(np.ceil(tracks.shape[1] / 16))
            if ((num_bars) % 2) == 1:    #pad zero so that each piece has a even number of bars (four beats per bar)
                pad_len = (num_bars + 1) * 16 - tracks.shape[1]
            else:
                pad_len = num_bars * 16 - tracks.shape[1]
            if pad_len != 0:
                tracks = np.pad(tracks, ((0, 0), (0, pad_len), (0, 0)), mode='constant', constant_values=(0,))

            center_pitch = compute_center_pitch(tracks)
            pitch_sort = np.argsort(center_pitch)[::-1]
            tracks = tracks[pitch_sort]
            programs = programs[pitch_sort]

            """clipping"""
            db_indices = np.nonzero(db_indicator)[0]
            if self.split == 'train':
                for i in range(0, len(db_indices), BAR_HOP_LEN):
                    if db_indices[i] + SAMPLE_LEN >= tracks.shape[1]:
                        break
                    self.anchor_list.append((len(self.pr_list), db_indices[i], tracks.shape[1]))  #(song_id, start, total_length)
            else:
                for i in range(0, tracks.shape[1], SAMPLE_LEN):
                    if i + SAMPLE_LEN >= tracks.shape[1]:
                        break
                    self.anchor_list.append((len(self.pr_list), i, tracks.shape[1]))  #(song_id, start, total_length)
            self.anchor_list.append((len(self.pr_list), max(0, (tracks.shape[1]-SAMPLE_LEN)), tracks.shape[1]))
            
            program_classes = self.slakh_program_mapping(programs)
            prog_sample = np.array([EMBED_PROGRAM_MAPPING[prog] for prog in program_classes])
            self.program_list.append(prog_sample)
            self.pr_list.append(tracks)


def collate_fn(batch, device, pitch_shift=True, get_pr_gt=False):
    max_dur = max([item[0].shape[1]//32 for item in batch])
    max_tracks = max([len(item[1]) for item in batch])

    grid_flatten_batch = []
    prog_batch = []
    time_mask = []
    track_mask = []
    func_pitch_batch = []
    func_time_batch = []
    total_length = []
    abs_pos = []
    rel_pos = []
    pr_batch = []

    if pitch_shift:
        aug_p = AUG_P / AUG_P.sum()
        aug_shift = np.random.choice(np.arange(-6, 6), 1, p=aug_p)[0]
    else:
        aug_shift = 0

    for pr, prog, (start, total_len) in batch:
        time_mask.append([0]*(pr.shape[1]//32) + [1]*(max_dur-pr.shape[1]//32))
        track_mask.append([0]*len(prog) + [1]*(max_tracks-len(prog)))

        r_pos = np.round(np.arange(start//32, (start+pr.shape[1])//32, 1) / (total_len//32-1) * len(REL_POS_BIN))
        total_len = np.argmin(np.abs(TOTAL_LEN_BIN - total_len//32)).repeat(pr.shape[1]//32)
        if start//32 <= ABS_POS_BIN[-2]:
            a_pos = np.append(ABS_POS_BIN[start//32: min(ABS_POS_BIN[-1], (start+pr.shape[1])//32)], [ABS_POS_BIN[-1]] * ((start+pr.shape[1])//32-ABS_POS_BIN[-1]))
        else:
            a_pos = np.array([ABS_POS_BIN[-1]] * (pr.shape[1]//32))


        pr = pr_mat_pitch_shift(pr, aug_shift)
        func_pitch, func_time = compute_pr_feat(pr.reshape(pr.shape[0], -1, 32, pr.shape[-1]))
        func_pitch = func_pitch.transpose(1, 0, 2)
        func_time = func_time.transpose(1, 0, 2)
        if len(prog) < max_tracks:
            pr = np.pad(pr, ((0, max_tracks-len(prog)), (0, 0), (0, 0)), mode='constant', constant_values=(0,))
            prog = np.pad(prog, ((0, max_tracks-len(prog))), mode='constant', constant_values=(NUM_INSTR_CLASS,))
            func_pitch = np.pad(func_pitch, ((0, 0), (0, max_tracks-func_pitch.shape[1]), (0, 0)), mode='constant', constant_values=(0,))
            func_time = np.pad(func_time, ((0, 0), (0, max_tracks-func_time.shape[1]), (0, 0)), mode='constant', constant_values=(0,))

        if pr.shape[1]//32 < max_dur:
            pr = np.pad(pr, ((0, 0), (0, max_dur*32-pr.shape[1]), (0, 0)), mode='constant', constant_values=(0,))
            total_len = np.pad(total_len, (0, max_dur-pr.shape[1]//32), mode='constant', constant_values=(len(TOTAL_LEN_BIN),))
            a_pos = np.pad(a_pos, (0, max_dur-len(a_pos)), mode='constant', constant_values=(len(ABS_POS_BIN),))
            r_pos = np.pad(r_pos, (0, max_dur-len(r_pos)), mode='constant', constant_values=(len(REL_POS_BIN),))
            func_pitch = np.pad(func_pitch, ((0, max_dur-len(func_pitch)), (0, 0)), mode='constant', constant_values=(0,))
            func_time = np.pad(func_time, ((0, max_dur-len(func_time)), (0, 0), (0, 0)), mode='constant', constant_values=(0,))

        #print('pr', pr.shape, 'prog', prog.shape, 'fp', func_pitch.shape, 'ft', func_time.shape)
        grid_flatten = pr2grid(np.max(pr, axis=0), max_note_count=32).reshape(-1, 32, 32, 6)
        grid_flatten_batch.append(grid_flatten)
        prog_batch.append(prog)
        func_pitch_batch.append(func_pitch)
        func_time_batch.append(func_time)
        total_length.append(total_len)
        abs_pos.append(a_pos)
        rel_pos.append(r_pos)
        pr_batch.append(pr)
        
    if get_pr_gt:
        return torch.from_numpy(np.array(pr_batch)).long().to(device), \
            torch.from_numpy(np.array(grid_flatten_batch)).long().to(device), \
            torch.from_numpy(np.array(prog_batch)).to(device), \
            torch.from_numpy(np.array(func_pitch_batch)).float().to(device), \
            torch.from_numpy(np.array(func_time_batch)).float().to(device), \
            torch.BoolTensor(time_mask).to(device), \
            torch.BoolTensor(track_mask).to(device), \
            torch.from_numpy(np.array(total_length)).long().to(device),\
            torch.from_numpy(np.array(abs_pos)).long().to(device),\
            torch.from_numpy(np.array(rel_pos)).long().to(device)
    else:
        return torch.from_numpy(np.array(grid_flatten_batch)).long().to(device), \
            torch.from_numpy(np.array(prog_batch)).to(device), \
            torch.from_numpy(np.array(func_pitch_batch)).float().to(device), \
            torch.from_numpy(np.array(func_time_batch)).float().to(device), \
            torch.BoolTensor(time_mask).to(device), \
            torch.BoolTensor(track_mask).to(device), \
            torch.from_numpy(np.array(total_length)).long().to(device),\
            torch.from_numpy(np.array(abs_pos)).long().to(device),\
            torch.from_numpy(np.array(rel_pos)).long().to(device)
        


def pr_mat_pitch_shift(pr_mat, shift):
    pr_mat = pr_mat.copy()
    pr_mat = np.roll(pr_mat, shift, -1)
    return pr_mat


def pr2grid(pr_mat, max_note_count=16, max_pitch=127, min_pitch=0,
                       pitch_pad_ind=130, dur_pad_ind=2,
                       pitch_sos_ind=128, pitch_eos_ind=129):
        grid = np.ones((SAMPLE_LEN, max_note_count, 6), dtype=int) * dur_pad_ind
        grid[:, :, 0] = pitch_pad_ind
        grid[:, 0, 0] = pitch_sos_ind
        cur_idx = np.ones(SAMPLE_LEN, dtype=int)
        for t, p in zip(*np.where(pr_mat != 0)):
            if cur_idx[t] == max_note_count - 1:
                continue
            grid[t, cur_idx[t], 0] = p - min_pitch
            binary = np.binary_repr(min(int(pr_mat[t, p]), 32) - 1, width=5)
            grid[t, cur_idx[t], 1: 6] = \
                np.fromstring(' '.join(list(binary)), dtype=int, sep=' ')
            cur_idx[t] += 1
        grid[np.arange(0, SAMPLE_LEN), cur_idx, 0] = pitch_eos_ind
        return grid


def compute_pr_feat(pr):
    #pr: (track, time, 128)
    onset = (np.sum(pr, axis=-1) > 0) * 1.   #(track, time)
    func_time = np.clip(np.sum((pr > 0) * 1., axis=-1) / 14, a_min=None, a_max=1)    #(track, time)
    func_pitch = np.sum((pr > 0) * 1., axis=-2) / 32
    
    return func_pitch, func_time

def compute_center_pitch(pr):
    #pr: (track, time, 128)
    #pr[pr > 0] = 1
    weight = np.sum(pr, axis=(-2, -1))
    weight[weight == 0] = 1
    pitch_center = np.sum(np.arange(0, 128)[np.newaxis, np.newaxis, :] * pr, axis=(-2, -1)) / weight 
    return pitch_center #(track, )