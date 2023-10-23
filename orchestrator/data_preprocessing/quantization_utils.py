import os
import numpy as np
import pretty_midi as pyd
from scipy.interpolate import interp1d

import sys
sys.path.append('../exported_midi_chord_recognition/')
from mir import DataEntry
from mir import io
from extractors.midi_utilities import get_valid_channel_count,is_percussive_channel,MidiBeatExtractor
from main import process_chord
import mir_eval

import librosa
from tqdm import tqdm

from joblib import Parallel, delayed
from multiprocessing import Manager


class Converter:
    def __init__(self, ACC):
        self.ACC = ACC
        self.max_note_count=11
        self.max_pitch=107
        self.min_pitch=22,
        self.pitch_sos_ind=86
        self.pitch_eos_ind=87
        self.pitch_pad_ind=88
        self.dur_pad_ind=2


    def midi2matrix(self, midi, quaver):
        """
        Convert multi-track midi to a 3D matrix of shape (Track, Time, 128). 
        Each cell is a integer number representing quantized duration.
        """
        tracks = []
        unique_programs = []
        for track in midi.instruments:  
            """
            merge tracks with the same program number into one single track
            """
            if not (track.program in unique_programs):
                unique_programs.append(track.program)
                tracks.append(track)
            else:
                idx = unique_programs.index(track.program)
                notes = tracks[idx].notes + track.notes
                notes = sorted(notes, key=lambda x: x.start, reverse=False)
                tracks[idx].notes = notes

        pr_matrices = []
        for track in tracks:
            pr_matrix = np.zeros((len(quaver), 128))
            for note in track.notes:
                note_start = np.argmin(np.abs(quaver - note.start))
                note_end =  np.argmin(np.abs(quaver - note.end))
                pr_matrix[note_start, note.pitch] = note_end - note_start
            pr_matrices.append(pr_matrix)
        return np.array(pr_matrices, dtype=np.uint8), np.array(unique_programs, dtype=np.uint8)

    def matrix_compress(self, pr_mat):
        #pr_mat: (time, 128)
        T = pr_mat.shape[0]
        pr_mat3d = np.zeros((T, self.max_note_count, 2), dtype=np.uint8)
        pr_mat3d[:, :, 0] = self.pitch_pad_ind
        pr_mat3d[:, 0, 0] = self.pitch_sos_ind
        cur_idx = np.ones(T, dtype=int)
        for t, p in zip(*np.where(pr_mat != 0)):
            if p < self.min_pitch or p > self.max_pitch:
                continue
            pr_mat3d[t, cur_idx[t], 0] = p - self.min_pitch
            pr_mat3d[t, cur_idx[t], 1] = min(int(pr_mat[t, p]), 128)
            if cur_idx[t] == self.max_note_count-1:
                continue
            cur_idx[t] += 1
        #print(cur_idx)
        pr_mat3d[np.arange(0, T), cur_idx, 0] = self.pitch_eos_ind
        return pr_mat3d

    def matrix_decompress(self, mat_compress):
        #mat_compress: (time, max_simu_note, 2)
        pr_mat = np.zeros((mat_compress.shape[0], 128))

        for t, p in zip(*np.where(mat_compress[:, 1:, 0] < self.pitch_eos_ind)):
            pitch_rel = mat_compress[t, p+1, 0]
            pr_mat[t, pitch_rel + self.min_pitch] = mat_compress[t, p+1, 1]

        return pr_mat


    def matrix2midi(self, pr_matrices, programs, init_tempo=120, time_start=0):
        """
        Reconstruct a multi-track midi from a 3D matrix of shape (Track. Time, 128).
        """
        tracks = []
        for program in programs:
            track_recon = pyd.Instrument(program=int(program), is_drum=False, name=pyd.program_to_instrument_name(int(program)))
            tracks.append(track_recon)

        indices_track, indices_onset, indices_pitch = np.nonzero(pr_matrices)
        alpha = 1 / (self.ACC // 4) * 60 / init_tempo #timetep between each quntization bin
        for idx in range(len(indices_track)):
            track_id = indices_track[idx]
            onset = indices_onset[idx]
            pitch = indices_pitch[idx]

            start = onset * alpha
            duration = pr_matrices[track_id, onset, pitch] * alpha
            velocity = 100

            note_recon = pyd.Note(velocity=int(velocity), pitch=int(pitch), start=time_start + start, end=time_start + start + duration)
            tracks[track_id].notes.append(note_recon)
        
        midi_recon = pyd.PrettyMIDI(initial_tempo=init_tempo)
        midi_recon.instruments = tracks
        return midi_recon


    def extrac_chord_matrix(self, midi_path, quaver, extra_division=1):
        '''
        Perform chord recognition on a midi
        '''
        entry = DataEntry()
        entry.append_file(midi_path,io.MidiIO, 'midi')
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


    def chord_matrix2midi(self, chord_matrix, init_tempo=120, time_start=0):
        alpha = 1 / (self.ACC // 4) * 60 / init_tempo * 4 #timetep between each quntization bin
        onset_or_rest = [i for i in range(1, len(chord_matrix)) if (chord_matrix[i] != chord_matrix[i-1]).any()]
        onset_or_rest = [0] + onset_or_rest
        onset_or_rest.append(len(chord_matrix))

        chordTrack = pyd.Instrument(program=0, is_drum=False, name='Chord')
        for idx, onset in enumerate(onset_or_rest[:-1]):
            chordset = [int(i) for i in chord_matrix[onset]]
            start = onset * alpha
            end = onset_or_rest[idx+1] * alpha
            root = chordset[0]
            chroma = chordset[1: 13]
            bass_rel = chordset[13]
            bass_bitmap = np.roll(chroma, shift=-root-bass_rel)
            bass = root + bass_rel

            if np.argmax(bass_bitmap) + bass >= 12:
                register = 3
            else:
                register = 4

            for entry in np.nonzero(bass_bitmap)[0]:
                pitch = register * 12 + bass + entry
                note = pyd.Note(velocity=100, pitch=int(pitch), start=time_start + start, end=time_start + end)
                chordTrack.notes.append(note)
        return chordTrack
        

    def pr2gird(self, compress_pr):
        #compress_pr: (time, max_simu_note, 2)
        pr_mat3d = np.ones((compress_pr.shape[0], self.max_note_count, 6), dtype=int) * self.dur_pad_ind
        pr_mat3d[:, :, 0] = self.pitch_pad_ind
        pr_mat3d[:, 0, 0] = self.pitch_sos_ind
        cur_idx = np.ones(compress_pr.shape[0], dtype=int)
        for t, p in zip(*np.where(compress_pr[:, 1:, 0] < self.pitch_eos_ind)):
            pitch_rel = compress_pr[t, p+1, 0]
            duration = compress_pr[t, p+1, 1]
            pr_mat3d[t, cur_idx[t], 0] = pitch_rel
            binary = np.binary_repr(min(duration, 32 - t) - 1, width=5)
            pr_mat3d[t, cur_idx[t], 1: 6] = np.fromstring(' '.join(list(binary)), dtype=int, sep=' ')
            if cur_idx[t] == self.max_note_count-1:
                continue
            cur_idx[t] += 1
        pr_mat3d[np.arange(0, compress_pr.shape[0]), cur_idx, 0] = self.pitch_eos_ind
        return pr_mat3d

    def grid2pr(self, grid):
        #grid: (time, max_simu_note, 6)
        if grid.shape[1] == self.max_note_count:
            grid = grid[:, 1:]
        pr = np.zeros((grid.shape[0], 128), dtype=int)
        for t in range(grid.shape[0]):
            for n in range(grid.shape[1]):
                note = grid[t, n]
                if note[0] == self.pitch_eos_ind:
                    break
                pitch = note[0] + self.min_pitch
                dur = int(''.join([str(_) for _ in note[1:]]), 2) + 1
                pr[t, pitch] = dur
        return pr

    def expand_chord(self, chord, shift, relative=False):
        # chord = np.copy(chord)
        root = (chord[0] + shift) % 12
        chroma = np.roll(chord[1: 13], shift)
        bass = (chord[13] + shift) % 12
        root_onehot = np.zeros(12)
        root_onehot[int(root)] = 1
        bass_onehot = np.zeros(12)
        bass_onehot[int(bass)] = 1
        if not relative:
            pass
        #     chroma = np.roll(chroma, int(root))
        # print(chroma)
        # print('----------')
        return np.concatenate([root_onehot, chroma, bass_onehot])

        