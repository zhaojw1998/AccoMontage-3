import os
import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.interpolate import interp1d
import pretty_midi as pyd
from scipy import stats as st

"""
    This script processes the POP909 Dataset into .npz files.
    Processed dataset is available at https://github.com/zhaojw1998/Query-and-reArrange/tree/main/data/POP909.
"""


def convert_pop909(melody, bridge, piano, beat):
    pr = np.zeros((3, len(beat)*4, 128, 2))
    for (sb, sq, sde, eb, eq, ede, p, v) in melody:
        assert sde==4
        assert ede==4
        s_ind = int(sb * sde + sq)
        e_ind = int(eb * ede + eq)
        p = int(p)
        pr[0, s_ind, p, 0] = e_ind - s_ind
        pr[0, s_ind, p, 1] = v
    for (sb, sq, sde, eb, eq, ede, p, v) in bridge:
        assert sde==4
        assert ede==4
        s_ind = int(sb * sde + sq)
        e_ind = int(eb * ede + eq)
        p = int(p)
        pr[1, s_ind, p, 0] = e_ind - s_ind
        pr[1, s_ind, p, 1] = v
    for (sb, sq, sde, eb, eq, ede, p, v) in piano:
        assert sde==4
        assert ede==4
        s_ind = int(sb * sde + sq)
        e_ind = int(eb * ede + eq)
        p = int(p)
        pr[2, s_ind, p, 0] = e_ind - s_ind
        pr[2, s_ind, p, 1] = v
    return pr


def midi2matrix(midi, quaver):
        pr_matrices = []
        programs = []
        quantization_error = []
        for track in midi.instruments:
            qt_error = [] # record quantization error
            pr_matrix = np.zeros((len(quaver), 128, 2))
            for note in track.notes:
                note_start = np.argmin(np.abs(quaver - note.start))
                note_end =  np.argmin(np.abs(quaver - note.end))
                if note_end == note_start:
                    note_end = min(note_start + 1, len(quaver) - 1) # guitar/bass plunk typically results in a very short note duration. These note should be quantized to 1 instead of 0.
                pr_matrix[note_start, note.pitch, 0] = note_end - note_start
                pr_matrix[note_start, note.pitch, 1] = note.velocity

                #compute quantization error. A song with very high error (e.g., triple-quaver songs) will be discriminated and therefore discarded.
                if note_end == note_start:
                    qt_error.append(np.abs(quaver[note_start] - note.start) / (quaver[note_start] - quaver[note_start-1]))
                else:
                    qt_error.append(np.abs(quaver[note_start] - note.start) / (quaver[note_end] - quaver[note_start]))
                
            control_matrix = np.ones((len(quaver), 128, 1)) * -1
            for control in track.control_changes:
                #if control.time < time_end:
                #    if len(quaver) == 0:
                #        continue
                control_time = np.argmin(np.abs(quaver - control.time))
                control_matrix[control_time, control.number, 0] = control.value

            pr_matrix = np.concatenate((pr_matrix, control_matrix), axis=-1)
            pr_matrices.append(pr_matrix)
            programs.append(track.program)
            quantization_error.append(np.mean(qt_error))

        return np.array(pr_matrices), programs, quantization_error


def retrieve_control(pop909_midi_dir, song, tracks):
    src_dir = os.path.join(pop909_midi_dir, song.split('.')[0], song.replace('.npz', '.mid'))
    src_midi = pyd.PrettyMIDI(src_dir)
    beats = src_midi.get_beats()
    beats = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))
    ACC = 4
    quantize = interp1d(np.array(range(0, len(beats))) * ACC, beats, kind='linear')
    quaver = quantize(np.array(range(0, (len(beats) - 1) * ACC)))

    pr_matrices, programs, _ = midi2matrix(src_midi, quaver)

    from_4_bin = np.nonzero(tracks[0, :, :, 0])
    from_midi = np.nonzero(pr_matrices[0, :, :, 0])

    mid_length = min(from_midi[1].shape[0], from_4_bin[1].shape[0])
    diff = from_midi[1][: mid_length] - from_4_bin[1][: mid_length]
    diff_avg = np.mean(diff)
    diff_std = np.std(diff)

    if diff_std > 0:
        diff_record = []
        for roll_idx in range(-32, 32):
            roll_pitches = np.roll(from_midi[1], shift=roll_idx, axis=0)
            diff = roll_pitches[abs(roll_idx): mid_length-abs(roll_idx)] - from_4_bin[1][abs(roll_idx): mid_length-abs(roll_idx)]
            diff_avg = np.mean(diff)
            diff_std = np.std(diff)
            diff_record.append((roll_idx, diff_avg, diff_std))
            diff_record = sorted(diff_record, key=lambda x: x[2])

        roll_idx_min = diff_record[0][0]
        roll_times = np.roll(from_midi[0], shift=roll_idx_min, axis=0)
        diff = roll_times[abs(roll_idx_min): mid_length-abs(roll_idx_min)] - from_4_bin[0][abs(roll_idx_min): mid_length-abs(roll_idx_min)]
    else:
        diff = from_midi[0][: mid_length] - from_4_bin[0][: mid_length]
    return pr_matrices[:, :, :, 2: 3], st.mode(diff).mode[0]

# To get the 4-bin quantized version, contact ziyu.wang @nyu.edu. Credit to [Wang et al., 2020] https://github.com/ZZWaang/polyphonic-chord-texture-disentanglement
pop909_4bin_dir = '/data1/zhaojw/Q&A/POP909-Dataset/quantization/POP09-PIANOROLL-4-bin-quantization/'
# Get raw MIDI of POP909 from https://github.com/music-x-lab/POP909-Dataset
pop909_midi_dir = '/data1/zhaojw/Q&A/POP909-Dataset/POP909/'
meta_info = pd.read_excel(os.path.join(pop909_midi_dir, 'index.xlsx'))
save_root = '/data1/zhaojw/Q&A/POP909-Dataset/quantization/4_bin_midi_quantization_with_dynamics_and_chord/'

for split in ['train', 'validation', 'test']:
    save_split = os.path.join(save_root, split)
    if not os.path.exists(save_split):
        os.makedirs(save_split)
    print(f'processing {split} set ...')

    pop909_list = os.listdir(pop909_4bin_dir)
    if split == 'train':
        pop909_list = pop909_list[: int(len(pop909_list)*.8)]
    elif split == 'validation':
        pop909_list = pop909_list[int(len(pop909_list)*.8): int(len(pop909_list)*.9)]
    elif split == 'test':
        pop909_list = pop909_list[int(len(pop909_list)*.9): ]
    
    for song in tqdm(pop909_list):
        song_meta = meta_info[meta_info.song_id == int(song.replace('.npz', ''))]
        num_beats = song_meta.num_beats_per_measure.values[0]
        num_quavers = song_meta.num_quavers_per_beat.values[0]
        if int(num_beats) == 3 or int(num_quavers) == 3:
            continue    #neglect pieces with triplet meters
        pop909_data = np.load(os.path.join(pop909_4bin_dir, song))
        beats = pop909_data['beat']
        melody= pop909_data['melody']
        bridge= pop909_data['bridge']
        piano= pop909_data['piano']
        
        tracks = convert_pop909(melody, bridge, piano, beats)
        
        

        track_control = np.ones((tracks.shape[0], tracks.shape[1], 128, 1)) * -1
        cc, shift = retrieve_control(pop909_midi_dir, song, tracks)
        if shift >= 0:
            track_control[:, : min(cc.shape[1] - shift, track_control.shape[1])] = cc[:, shift: min(cc.shape[1], track_control.shape[1] + shift)]    
        else:
            track_control[:, -shift: min(cc.shape[1] - shift, track_control.shape[1]) ] = cc[:, :min(cc.shape[1], track_control.shape[1] + shift)]

        pr_matrices = tracks[..., 0]
        dynamic_matrices = np.concatenate([tracks[..., 1:], track_control], axis=-1)
        chord_matrices = pop909_data['chord']
        downbeat_indicator = np.zeros(len(beats)*4)
        for idx, beat in enumerate(beats):
            if beat[3] == 0:
                downbeat_indicator[idx*4] = 1

        #print(pr_matrices.shape)
        #print(dynamic_matrices.shape)
        #print(chord_matrices.shape)
        #print(downbeat_indicator.shape)

        np.savez(os.path.join(save_split, song),\
                    tracks = pr_matrices,\
                    db_indicator = downbeat_indicator,\
                    dynamics = dynamic_matrices, \
                    chord = chord_matrices)

        #break