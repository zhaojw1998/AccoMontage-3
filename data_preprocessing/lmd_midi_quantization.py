import os
import numpy as np
import pretty_midi as pyd
from tqdm import tqdm
from scipy.interpolate import interp1d
import yaml

import sys
sys.path.append('piano_arranger/chord_recognition')
from mir import DataEntry
from mir import io
from extractors.midi_utilities import MidiBeatExtractor
from main import process_chord
import mir_eval

"""
    This script processes the Lakh MIDI Dataset into .npz files.
"""


def midi2matrix(midi, quaver):
        """
        Convert multi-track midi to a 3D matrix of shape (Track, Time, 128). 
        Each cell is a integer number representing quantized duration.
        """
        pr_matrices = []
        programs = []
        quantization_error = []
        for track in midi.instruments:
            if track.is_drum:
                continue
            qt_error = [] # record quantization error
            pr_matrix = np.zeros((len(quaver), 128))
            for note in track.notes:
                note_start = np.argmin(np.abs(quaver - note.start))
                note_end =  np.argmin(np.abs(quaver - note.end))
                if note_end == note_start:
                    note_end = min(note_start + 1, len(quaver) - 1) # guitar/bass plunk typically results in a very short note duration. These note should be quantized to 1 instead of 0.
                pr_matrix[note_start, note.pitch] = note_end - note_start

                #compute quantization error. A song with very high error (e.g., triple-quaver songs) will be discriminated and therefore discarded.
                if note_end == note_start:
                    qt_error.append(np.abs(quaver[note_start] - note.start) / (quaver[note_start] - quaver[note_start-1]))
                else:
                    qt_error.append(np.abs(quaver[note_start] - note.start) / (quaver[note_end] - quaver[note_start]))
            
            pr_matrices.append(pr_matrix)
            programs.append(track.program)
            quantization_error.append(np.mean(qt_error))

        return np.array(pr_matrices), np.array(programs), quantization_error


def extrac_chord_matrix(midi_path, quaver, extra_division=1):
        '''
        Perform chord recognition on a midi
        '''
        entry = DataEntry()
        entry.append_file(midi_path, io.MidiIO, 'midi')
        entry.append_extractor(MidiBeatExtractor, 'beat')
        result = process_chord(entry, extra_division)

        beat_quaver = quaver[::4]
        chord_matrix = np.zeros((len(beat_quaver), 14))
        chord_matrix[:, 0] = -1
        chord_matrix[:, -1] = -1

        for chord in result:
            chord_start = np.argmin(np.abs(beat_quaver - chord[0]))
            chord_end =  np.argmin(np.abs(beat_quaver - chord[1]))
            root, bitmap, bass_rel = mir_eval.chord.encode(chord[2])
            chroma = mir_eval.chord.rotate_bitmap_to_root(bitmap, root)
            chord = np.concatenate(([root], chroma, [bass_rel]), axis=-1)
            chord_matrix[chord_start: chord_end, :] = chord
        return chord_matrix


ACC = 4
#We want to exclude from LMD the pieces included in Slkh2100
slakh_root = '/data1/zhaojw/Q&A/slakh2100_flac_redux/'  #Download slakh2100_flac_redux from https://zenodo.org/records/4599666
slakh_ids = []
for split in ['train', 'validation', 'test', 'omitted']:
    slakh_split = os.path.join(slakh_root, split)
    for song in tqdm(os.listdir(slakh_split)):
        track_id = yaml.safe_load(open(os.path.join(slakh_split, song, 'metadata.yaml'), 'r'))['UUID']
        slakh_ids.append(track_id)
print(len(slakh_ids))

lmd_root = '/data1/zhaojw/LMD/lmd_full/'  #Download lmd_full from https://colinraffel.com/projects/lmd/#get
lmd_midi = {}
slakh_midi = {}
for folder in os.listdir(lmd_root):
    sub_folder = os.path.join(lmd_root, folder)
    for piece in os.listdir(sub_folder):
        midi_id = piece.split('.')[0]
        if midi_id in slakh_ids:
            slakh_midi[midi_id] = os.path.join(sub_folder, piece)
        else:
            lmd_midi[midi_id] = os.path.join(sub_folder, piece)
print(len(slakh_midi), len(lmd_midi))


save_root = "/data1/zhaojw/LMD/4_bin_quantization/"
print(f'processing LMD ...')
for song_id in tqdm(lmd_midi):
    break_flag = 0

    try:
        all_src_midi = pyd.PrettyMIDI(lmd_midi[song_id])
    except:
        continue
    for ts in all_src_midi.time_signature_changes:
        if not (((ts.numerator == 2) or (ts.numerator == 4)) and (ts.denominator == 4)):
            break_flag = 1
            break
    if break_flag:
        continue    # process only 2/4 and 4/4 songs

    beats = all_src_midi.get_beats()
    if len(beats) < 32:
        continue    #skip pieces shorter than 8 bars
    downbeats = all_src_midi.get_downbeats()

    beats = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))
    quantize = interp1d(np.array(range(0, len(beats))) * ACC, beats, kind='linear')
    quaver = quantize(np.array(range(0, (len(beats) - 1) * ACC)))
    
    break_flag = 0
    pr_matrices, programs, track_qt = midi2matrix(all_src_midi, quaver)
    for item in track_qt:
        if item > .2:
            break_flag = 1
    if break_flag:
        continue    #skip the pieces with very large quantization error. This pieces are possibly triple-quaver songs
    db_indicator = np.array([int(t in downbeats) for t in quaver])

    np.savez(os.path.join(save_root, f'{song_id}.npz'),\
                    tracks = pr_matrices,\
                    programs = programs,\
                    db_indicator = db_indicator)
    
    
    """save_root = "/data1/zhaojw/LMD/4_bin_quantization_chord/"
    try:
        chord_matrix = extrac_chord_matrix(lmd_midi[song_id], quaver)
    except:
        continue
    
    np.save(os.path.join(save_root, f'{song_id}.npy'), chord_matrix)"""

        