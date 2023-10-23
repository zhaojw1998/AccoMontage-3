import os
import numpy as np
import pretty_midi as pyd
import soundfile as sf
from tqdm import tqdm
from math import ceil
from scipy.interpolate import interp1d
from quantization_utils import Converter
import librosa
import yaml

import sys
sys.path.append('../exported_midi_chord_recognition')
from mir import DataEntry
from mir import io
from extractors.midi_utilities import get_valid_channel_count,is_percussive_channel,MidiBeatExtractor
from main import process_chord
import mir_eval

from time_stretch import time_stretch


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


TGT_SR = 22050
HOP_LGTH = 512
STRETCH_BPM = 100
AUD_DIR = '../slakh2100_flac_redux/'

def pad_audio_npy(audio, beat_secs, exceed_frames=1000):
    """
    This operation generates a copy of the wav and ensures
    len(copy) >= frame of beat_secs[-1] * TGT_SR + exceed_frames
    """
    last_beat_frame = beat_secs[-1] * TGT_SR
    last_audio_frame = len(audio) - 1
    if last_audio_frame < last_beat_frame + exceed_frames:
        pad_data = np.zeros(ceil(last_beat_frame + exceed_frames),
                            dtype=np.float32)
        pad_data[0: len(audio)] = audio
    else:
        pad_data = audio.copy()
    return pad_data


def stretch_a_song(beat_secs, audio, tgt_bpm=100, exceed_frames=1000):
    """Stretch the audio to constant bpm=tgt_bpm."""
    data = pad_audio_npy(audio, beat_secs, exceed_frames=exceed_frames)
    pad_start = 0
    if beat_secs[0] > HOP_LGTH / TGT_SR:
        critical_beats = np.insert(beat_secs, 0, 0)
        beat_dict = dict(zip(beat_secs,
                             np.arange(0, len(beat_secs)) + 1))
        pad_start = 1
    else:
        critical_beats = beat_secs
        beat_dict = dict(zip(beat_secs,
                             np.arange(0, len(beat_secs))))

    critical_frames = critical_beats * TGT_SR
    critical_frames = np.append(critical_frames, len(data))

    frame_intervals = np.diff(critical_frames)
    tgt_interval = (60 / tgt_bpm) * TGT_SR
    rates = frame_intervals / tgt_interval

    steps = [np.arange(critical_frames[i] / HOP_LGTH,
                       critical_frames[i + 1] / HOP_LGTH,
                       rates[i])
             for i in range(len(frame_intervals))]

    time_steps = np.concatenate(steps, dtype=float)

    fpb = np.ceil((tgt_interval / HOP_LGTH)) * HOP_LGTH
    len_stretch = int(fpb * len(steps))

    stretched_song = time_stretch(data, time_steps, len_stretch,
                                  center=False)
    beat_steps = [int(i * fpb) for i in range(len(steps))]
    if pad_start:
        beat_steps = beat_steps[1:]
    return stretched_song, beat_steps, int(fpb), rates



ACC = 4

slakh_root = '../slakh2100_flac_redux'
save_root = '../slakh2100_flac_redux/4_bin_quantization/'
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

        audio, _ = librosa.load(os.path.join(slakh_split, song, 'drum_detach', 'drum_detach_22050.wav'), sr=TGT_SR)
        audio_strech, beat_steps, fpb, _ = stretch_a_song(beats[:-1], audio, tgt_bpm=STRETCH_BPM)
        sf.write(os.path.join(slakh_split, song, 'drum_detach', f'drum_detach_22050_{STRETCH_BPM}.wav'), audio_strech, TGT_SR, 'PCM_16')


        db_indicator = np.array([int(t in downbeats) for t in quaver], dtype=np.uint8)
        db_frame = [beat_steps[i] for i in range(len(beats[:-1])) if beats[i] in downbeats]

        assert(len(np.nonzero(db_indicator)[0]) == len(db_frame))

        #print(db_frame)
        #print(len(db_frame), len(np.nonzero(db_indicator)[0]))

        #print(beat_steps)
        #print(len(beat_steps), len(downbeats))
        #print(fpb)

        np.savez(os.path.join(save_split, f'{song}.npz'),\
                    tracks = pr_matrices,\
                    programs = programs,\
                    chord = chord_matrix,\
                    db_indicator = db_indicator,\
                    db_frame = db_frame,\
                    fpb = fpb)

    