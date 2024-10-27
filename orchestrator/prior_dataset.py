import os
import numpy as np
import pretty_midi as pyd
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

SAMPLE_LEN = 16 #16 codes per sample sequence, where each codes represents 2-bar segment
HOP_LEN = 4
NUM_INSTR_CLASS = 34
NUM_TIME_CODE = 128
TOTAL_LEN_BIN = np.array([4, 7, 12, 15, 20, 23, 28, 31, 36, 39, 44, 47, 52, 55, 60, 63, 68, 71, 76, 79, 84, 87, 92, 95, 100, 103, 108, 111, 116, 119, 124, 127, 132])
ABS_POS_BIN = np.arange(129)
REL_POS_BIN = np.arange(128)


class VQ_LMD_Dataset(Dataset):
    def __init__(self, lmd_dir, debug_mode=False, split='train', mode='train'):
        super(VQ_LMD_Dataset, self).__init__()
        self.lmd_dir = lmd_dir
        self.split = split
        self.mode = mode
        self.debug_mode = debug_mode
        self.reduction_list = []
        self.program_list = [] 
        self.function_list = []
        self.anchor_list = []
        
        print('loading LMD Dataset ...')
        self.load_lmd()

    def __len__(self):
        return len(self.anchor_list)
    
    def __getitem__(self, idx):
        song_id, start, total_len = self.anchor_list[idx]
        pno_red = self.reduction_list[song_id][start: min(total_len, start+SAMPLE_LEN)]
        prog = self.program_list[song_id]
        function = self.function_list[song_id][start: min(total_len, start+SAMPLE_LEN)]
        return pno_red, prog, function, (start, total_len)



    def load_lmd(self):
        lmd_list = os.listdir(self.lmd_dir)
        if self.split == 'train':
            lmd_list = lmd_list[: int(len(lmd_list)*.95)]
        elif self.split == 'validation':
            lmd_list = lmd_list[int(len(lmd_list)*.95): ]
        if self.debug_mode:
            lmd_list = lmd_list[: 1000]
        for song in tqdm(lmd_list):
            lmd_data = np.load(os.path.join(self.lmd_dir, song))
            pno_red = lmd_data['mixture']   #(num2bar, 256)    3 for duration, velocity, and control
            prog = lmd_data['programs']    #(track)
            if len(prog) > 20:
                continue    #for sake of computing memory
            function = lmd_data['func_time']    #(num2bar, n_track, 8)

            if self.split == 'train':
                for i in range(0, len(pno_red), HOP_LEN):
                    if i + SAMPLE_LEN >= len(pno_red):
                        break
                    self.anchor_list.append((len(self.reduction_list), i, len(pno_red)))  #(song_id, start, total_length)
            else:
                for i in range(0, len(pno_red), SAMPLE_LEN):
                    if i + SAMPLE_LEN >= len(pno_red):
                        break
                    self.anchor_list.append((len(self.reduction_list), i, len(pno_red)))  #(song_id, start, total_length)
            self.anchor_list.append((len(self.reduction_list), max(0, len(pno_red)-SAMPLE_LEN), len(pno_red)))

            self.reduction_list.append(pno_red)
            self.program_list.append(prog)
            self.function_list.append(function)



def collate_fn(batch, device):
    max_dur = max([len(item[0]) for item in batch])
    max_tracks = max([len(item[1]) for item in batch])

    reduction = []
    programs = []
    function = []
    time_mask = []
    track_mask = []
    total_length = []
    abs_pos = []
    rel_pos = []

    for pno_red, prog, func, (start, total_len) in batch:
        time_mask.append([0]*len(pno_red) + [1]*(max_dur-len(pno_red)))
        track_mask.append([0]*len(prog) + [1]*(max_tracks-len(prog)))
        
        r_pos = np.round(np.arange(start, start+len(pno_red), 1) / (total_len-1) * len(REL_POS_BIN))
        total_len = np.argmin(np.abs(TOTAL_LEN_BIN - total_len)).repeat(len(pno_red))
        if start <= ABS_POS_BIN[-2]:
            a_pos = np.append(ABS_POS_BIN[start: min(ABS_POS_BIN[-1], start+len(pno_red))], [ABS_POS_BIN[-1]] * (start+len(pno_red)-ABS_POS_BIN[-1]))
        else:
            a_pos = np.array([ABS_POS_BIN[-1]] * len(pno_red))

        a = np.random.rand()
        if a < 0.3:
            blur_ratio = 0
        elif a < 0.7:
            blur_ratio = (np.random.rand() * 2 + 1) / 4 #range in [.25, .75)
        else:
            blur_ratio = 1
        pno_red = (1 - blur_ratio) * pno_red + blur_ratio * np.random.normal(loc=0, scale=1, size=pno_red.shape)

        if len(prog) < max_tracks:
            prog = np.pad(prog, ((0, max_tracks-len(prog))), mode='constant', constant_values=(NUM_INSTR_CLASS,))
            func = np.pad(func, ((0, 0), (0, max_tracks-func.shape[1]), (0, 0)), mode='constant', constant_values=(NUM_TIME_CODE,))

        if len(pno_red) < max_dur:
            pno_red = np.pad(pno_red, ((0, max_dur-len(pno_red)), (0, 0)), mode='constant', constant_values=(0,))
            total_len = np.pad(total_len, (0, max_dur-len(total_len)), mode='constant', constant_values=(len(TOTAL_LEN_BIN),))
            a_pos = np.pad(a_pos, (0, max_dur-len(a_pos)), mode='constant', constant_values=(len(ABS_POS_BIN),))
            r_pos = np.pad(r_pos, (0, max_dur-len(r_pos)), mode='constant', constant_values=(len(REL_POS_BIN),))
            func = np.pad(func, ((0, max_dur-len(func)), (0, 0), (0, 0)), mode='constant', constant_values=(NUM_TIME_CODE,))
        
        reduction.append(pno_red)
        programs.append(prog)
        function.append(func)
        total_length.append(total_len)
        abs_pos.append(a_pos)
        rel_pos.append(r_pos)
        
    return torch.from_numpy(np.array(reduction)).float().to(device), \
            torch.from_numpy(np.array(programs)).long().to(device), \
            torch.from_numpy(np.array(function)).long().to(device), \
            torch.BoolTensor(time_mask).to(device), \
            torch.BoolTensor(track_mask).to(device), \
            torch.from_numpy(np.array(total_length)).long().to(device), \
            torch.from_numpy(np.array(abs_pos)).long().to(device), \
            torch.from_numpy(np.array(rel_pos)).long().to(device)



if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES']= '0'
    from torch.utils.data import DataLoader
    DEVICE = 'cuda:0'
    dataset = VQ_LMD_Dataset(lmd_dir="/data1/zhaojw/LMD/4_bin_quantization_VQ", debug_mode=True, split='Train', mode='train')
    loader = DataLoader(dataset, batch_size=128, shuffle=True, collate_fn=lambda b: collate_fn(b, DEVICE))

    for pno_red, programs, function, track_mask, time_mask, total_length, abs_pos, rel_pos in loader:
        print(pno_red.shape)
        print(programs.shape)
        print(function.shape)
        print(track_mask.shape)
        print(time_mask.shape)
        print(total_length.shape)
        print(abs_pos.shape)
        print(rel_pos.shape)
        break