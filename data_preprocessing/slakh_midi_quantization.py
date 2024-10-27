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
    This script processes the Slakh2100 Dataset (MIDI) into .npz files.
    Processed dataset is available at https://github.com/zhaojw1998/Query-and-reArrange/tree/main/data/Slakh2100.
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

        return np.array(pr_matrices, dtype=np.uint8), programs, quantization_error


def extrac_chord_matrix(midi_path, quaver, extra_division=1):
        '''
        Perform chord recognition on a midi
        '''
        entry = DataEntry()
        entry.append_file(midi_path, io.MidiIO, 'midi')
        entry.append_extractor(MidiBeatExtractor, 'beat')
        result = process_chord(entry, extra_division)

        beat_quaver = quaver[::4]
        chord_matrix = np.zeros((len(beat_quaver), 14), dtype=np.uint8)
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

slakh_root = '/data1/zhaojw/Q&A/slakh2100_flac_redux'  #Download slakh2100_flac_redux from https://zenodo.org/records/4599666
save_root = '/data1/zhaojw/Q&A/slakh2100_flac_redux/4_bin_quantization/'
for split in ['train', 'validation', 'test']:
    slakh_split = os.path.join(slakh_root, split)
    save_split = os.path.join(save_root, split)
    if not os.path.exists(save_split):
        os.mkdir(save_split)
    print(f'processing {split} set ...')
    for song in tqdm(os.listdir(slakh_split)):
        break_flag = 0

        all_src_midi = pyd.PrettyMIDI(os.path.join(slakh_split, song, 'all_src.mid'))
        for ts in all_src_midi.time_signature_changes:
            if not (((ts.numerator == 2) or (ts.numerator == 4)) and (ts.denominator == 4)):
                break_flag = 1
                break
        if break_flag:
            continue    # process only 2/4 and 4/4 songs

        tracks = os.path.join(slakh_split, song, 'MIDI')
        track_names = os.listdir(tracks)
        track_midi = [pyd.PrettyMIDI(os.path.join(tracks, track)) for track in track_names]
        track_meta = yaml.safe_load(open(os.path.join(slakh_split, song, 'metadata.yaml'), 'r'))['stems']

        if len(all_src_midi.get_beats()) >= max([len(midi.get_beats()) for midi in track_midi]):
            beats = all_src_midi.get_beats()
            downbeats = all_src_midi.get_downbeats()
        else:
            beats = track_midi[np.argmax([len(midi.get_beats()) for midi in track_midi])].get_beats()
            downbeats = track_midi[np.argmax([len(midi.get_beats()) for midi in track_midi])].get_downbeats()

        beats = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))
        quantize = interp1d(np.array(range(0, len(beats))) * ACC, beats, kind='linear')
        quaver = quantize(np.array(range(0, (len(beats) - 1) * ACC)))
        
        pr_matrices = []
        programs = []
        
        break_flag = 0
        for idx, midi in enumerate(track_midi):
            meta = track_meta[track_names[idx].replace('.mid', '')]
            if meta['is_drum']:
                continue    #let's skip drum for now
            pr_matrix, _, track_qt = midi2matrix(midi, quaver)
            if track_qt[0] > .2:
                break_flag = 1
                break
            pr_matrices.append(pr_matrix)
            programs.append(meta['program_num'])
        if break_flag:
            continue    #skip the pieces with very large quantization error. This pieces are possibly triple-quaver songs
        
        pr_matrices = np.concatenate(pr_matrices, axis=0, dtype=np.uint8)
        programs = np.array(programs, dtype=np.uint8)
        
        chord_matrix = extrac_chord_matrix(os.path.join(slakh_split, song, 'all_src.mid'), quaver)

        db_indicator = np.array([int(t in downbeats) for t in quaver], dtype=np.uint8)

        np.savez(os.path.join(save_split, f'{song}.npz'),\
                    tracks = pr_matrices,\
                    programs = programs,\
                    chord = chord_matrix,\
                    db_indicator = db_indicator)

    