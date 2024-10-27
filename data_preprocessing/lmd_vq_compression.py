import os
os.environ['CUDA_VISIBLE_DEVICES']= '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from orchestrator.autoencoder import Query_and_reArrange
from orchestrator.autoencoder_dataset import SAMPLE_LEN, SLAKH_PROGRAM_MAPPING, EMBED_PROGRAM_MAPPING

"""
    This script uses a pretrained autoencoder to encode Lakh MIDI Dataset into discrete latent codes.
    Processed dataset is available at .
"""


DEVICE = 'cuda:0'
autoencoder = Query_and_reArrange(name='pretrain', trf_layers=2, device=DEVICE)
autoencoder.load_state_dict(torch.load("/data1/zhaojw/data_file_dir/params_autoencoder.pt"))
autoencoder.to(DEVICE)
autoencoder.eval()

SLAKH_CLASS_MAPPING = {v: k for k, v in EMBED_PROGRAM_MAPPING.items()}


class LMD_Dataset(Dataset):
    def __init__(self, lmd_dir, debug_mode=False, split='train'):
        super(LMD_Dataset, self).__init__()
        self.lmd_dir = lmd_dir
        self.split = split
        self.debug_mode = debug_mode
       
        self.mix_list = []
        #self.fp_list = [] 
        self.ft_list = []
        self.prog_list = []

        print('loading LMD Dataset ...')
        self.load_lmd()

    def slakh_program_mapping(self, programs):
        return np.array([SLAKH_PROGRAM_MAPPING[program] for program in programs])


    def load_lmd(self):
        lmd_list = os.listdir(self.lmd_dir)
        if self.split == 'train':
            lmd_list = lmd_list[:len(lmd_list)*0.9]
        elif self.split == 'validation':
            lmd_list = lmd_list[-len(lmd_list)*0.9:]
        elif self.split == 'all':
            lmd_list = lmd_list
        if self.debug_mode:
            lmd_list = lmd_list[:10]
        #with open("/data1/zhaojw/LMD/melody_dict_exclude_drum.json", 'r') as f:
        #    self.melody_check = json.load(f)

        for song in tqdm(lmd_list):
            try:
                self.clip_one_piece(song)
            except RuntimeError:
                continue


    def clip_one_piece(self, song):
        #print(song)
        lmd_data = np.load(os.path.join(self.lmd_dir, song), allow_pickle=True)
        tracks = lmd_data['tracks']   #(n_track, time, 128)
        programs = lmd_data['programs']    #(n_track, )
        #db_indicator = lmd_data['db_indicator']  #(time, )

        #pr_recon = matrix2midi(tracks, programs, 100)
        #pr_recon.write(f'/home/zhaojw/workspace/workspace/AccoMontage3/prior_model_final/recon.mid')
        
        rare_sound_effect_tracks = np.flatnonzero(programs > 95)
        if len(rare_sound_effect_tracks) > 0:
            tracks = np.delete(tracks, rare_sound_effect_tracks, axis=0)
            programs = np.delete(programs, rare_sound_effect_tracks, axis=0)
        if len(tracks) == 0:
            return
        
        program_classes = self.slakh_program_mapping(programs)
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
        program_classes = program_classes[pitch_sort]
        #print(program_classes)

        tracks = tracks.reshape(tracks.shape[0], -1, 32, 128).transpose(1, 0, 2, 3) #(num2bar, n_track, 32, 128)
        _, ft = compute_pr_feat(tracks)
        

        ft_shape = ft.shape
        ft = ft.reshape(-1, 32)
        ft = autoencoder.function_enc.get_code_indices(torch.from_numpy(ft).to(DEVICE).float()).cpu().detach().numpy().reshape(*ft_shape[:-1], 8)
        #ft = ft.astype(np.uint)

        num_tracks = tracks.shape[1]
        if num_tracks > 3:
            max_discard = num_tracks//2 #discard half tracks at maximum
            num_discard = np.random.randint(low=0, high=max_discard+1)  #number of tracks to discard
            discard = np.random.choice(a=list(range(num_tracks)), size=num_discard, replace=False)
            tracks = np.delete(tracks, discard, axis=1)

        mix = np.array([pr2grid(matrix, max_note_count=32) for matrix in np.sum(tracks, axis=1)])
        mix, _, _ = autoencoder.piano_enc(torch.from_numpy(mix).to(DEVICE).long())
        mix = mix.mean.cpu().detach().numpy()
        
        prog = np.array([EMBED_PROGRAM_MAPPING[p] for p in program_classes])
        #prog = prog.astype(np.uint)

        save_root = "/data1/zhaojw/LMD/VQ-Q&A-T-009-reorder-random_discard/"
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        np.savez(os.path.join(save_root, song),\
                            mixture = mix,\
                            programs = prog,\
                            func_time = ft)


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
    #func_pitch = np.sum((pr > 0) * 1., axis=-2) / 32
    func_pitch = None
    
    return func_pitch, func_time

def compute_center_pitch(pr):
    #pr: (track, time, 128)
    #pr[pr > 0] = 1
    weight = np.sum(pr, axis=(-2, -1))
    weight[weight == 0] = 1
    pitch_center = np.sum(np.arange(0, 128)[np.newaxis, np.newaxis, :] * pr, axis=(-2, -1)) / weight 
    return pitch_center #(track, )


if __name__ == '__main__':
    dataset = LMD_Dataset(lmd_dir="/data1/zhaojw/LMD/4_bin_quantization/", debug_mode=False, split='all')