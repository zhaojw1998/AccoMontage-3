import os
import datetime
import shutil
import torch
import numpy as np
from torch.distributions import Normal, kl_divergence
from torch.utils.tensorboard import SummaryWriter
import pretty_midi as pyd
from scipy.interpolate import interp1d
from scipy import stats as st




def get_zs_from_dists(dists, sample=False):
    return [dist.rsample() if sample else dist.mean for dist in dists]


def scheduled_sampling(i, high=0.7, low=0.05, scaler=1e5):
    x = 10 * (i - 0.5)
    z = 1 / (1 + np.exp(x))
    y = (high - low) * z + low
    return y


def kl_anealing(i, high=0.1, low=0., scaler=None):
    hh = 1 - low
    ll = 1 - high
    x = 10 * (i - 0.5)
    z = 1 / (1 + np.exp(x))
    y = (hh - ll) * z + ll
    return 1 - y

def standard_normal(shape, device):
    N = Normal(torch.zeros(shape), torch.ones(shape))
    #if torch.cuda.is_available():
    N.loc = N.loc.to(device)
    N.scale = N.scale.to(device)
    return N

def kl_with_normal(dist):
    shape = dist.mean.size(-1)
    normal = standard_normal(shape, dist.mean.device)
    kl = kl_divergence(dist, normal).mean()
    return kl


class SummaryWriters:

    def __init__(self, writer_names, tags, log_path, tasks=('train', 'val')):
        # writer_names example: ['loss', 'kl_loss', 'recon_loss']
        # tags example: {'name1': None, 'name2': (0, 1)}
        self.log_path = log_path
        #assert 'loss' == writer_names[0]
        self.writer_names = writer_names
        self.tags = tags
        self._regularize_tags()

        writer_dic = {}
        for name in writer_names:
            writer_dic[name] = SummaryWriter(os.path.join(log_path, name))
        self.writers = writer_dic

        all_tags = {}
        for task in tasks:
            task_dic = {}
            for key, val in self.tags.items():
                task_dic['_'.join([task, key])] = val
            all_tags[task] = task_dic
        self.all_tags = all_tags

    def _init_summary_writer(self):
        tags = {'batch_train': (0, 1, 2, 3, 4)}
        self.summary_writers = SummaryWriters(self.writer_names, tags,
                                              self.writer_path)

    def _regularize_tags(self):
        for key, val in self.tags.items():
            if val is None:
                self.tags[key] = tuple(range(len(self.writer_names)))

    def single_write(self, name, tag, val, step):
        self.writers[name].add_scalar(tag, val, step)

    def write_tag(self, task, tag, vals, step):
        assert len(vals) == len(self.all_tags[task][tag])
        for name_id, val in zip(self.all_tags[task][tag], vals):
            name = self.writer_names[name_id]
            self.single_write(name, tag, val, step)

    def write_task(self, task, vals_dic, step):
        for tag, name_ids in self.all_tags[task].items():
            vals = [vals_dic[self.writer_names[i]] for i in name_ids]
            self.write_tag(task, tag, vals, step)


def join_fn(*items, ext='pt'):
    return '.'.join(['_'.join(items), ext])


class LogPathManager:

    def __init__(self, readme_fn=None, save_root='.', log_path_name='result',
                 with_date=True, with_time=True,
                 writer_folder='writers', model_folder='models'):
        date = str(datetime.date.today()) if with_date else ''
        ctime = datetime.datetime.now().time().strftime("%H%M%S") \
            if with_time else ''
        log_folder = '_'.join([date, ctime, log_path_name])
        log_path = os.path.join(save_root, log_folder)
        writer_path = os.path.join(log_path, writer_folder)
        model_path = os.path.join(log_path, model_folder)
        self.log_path = log_path
        self.writer_path = writer_path
        self.model_path = model_path
        LogPathManager.create_path(log_path)
        LogPathManager.create_path(writer_path)
        LogPathManager.create_path(model_path)
        if readme_fn is not None:
            shutil.copyfile(readme_fn, os.path.join(log_path, 'readme.txt'))

    @staticmethod
    def create_path(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def epoch_model_path(self, model_name):
        model_fn = join_fn(model_name, 'epoch', ext='pt')
        return os.path.join(self.model_path, model_fn)

    def valid_model_path(self, model_name):
        model_fn = join_fn(model_name, 'valid', ext='pt')
        return os.path.join(self.model_path, model_fn)

    def final_model_path(self, model_name):
        model_fn = join_fn(model_name, 'final', ext='pt')
        return os.path.join(self.model_path, model_fn)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def matrix2midi(pr_matrices, programs, init_tempo=120, time_start=0):
        """
        Reconstruct a multi-track midi from a 3D matrix of shape (Track. Time, 128).
        """
        ACC = 16
        tracks = []
        for program in programs:
            track_recon = pyd.Instrument(program=int(program), is_drum=False, name=pyd.program_to_instrument_name(int(program)))
            tracks.append(track_recon)

        indices_track, indices_onset, indices_pitch = np.nonzero(pr_matrices)
        alpha = 1 / (ACC // 4) * 60 / init_tempo #timetep between each quntization bin
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

def grid2pr(grid, max_note_count=16, min_pitch=0, pitch_eos_ind=129):
        #grid: (time, max_simu_note, 6)
        if grid.shape[1] == max_note_count:
            grid = grid[:, 1:]
        pr = np.zeros((grid.shape[0], 128), dtype=int)
        for t in range(grid.shape[0]):
            for n in range(grid.shape[1]):
                note = grid[t, n]
                if note[0] == pitch_eos_ind:
                    break
                pitch = note[0] + min_pitch
                dur = int(''.join([str(_) for _ in note[1:]]), 2) + 1
                pr[t, pitch] = dur
        return pr

def midi2matrix(midi, quaver):
    pr_matrices = []
    programs = []
    for track in midi.instruments:
        programs.append(track.program)
        pr_matrix = np.zeros((len(quaver), 128))
        for note in track.notes:
            note_start = np.argmin(np.abs(quaver - note.start))
            note_end =  np.argmin(np.abs(quaver - note.end))
            if note_end == note_start:
                note_end = min(note_start + 1, len(quaver) - 1)
            pr_matrix[note_start, note.pitch] = note_end - note_start
        pr_matrices.append(pr_matrix)
    return np.array(pr_matrices), np.array(programs)


def midi2matrix_with_dynamics(midi, quaver):
        """
        Convert multi-track midi to a 3D matrix of shape (Track, Time, 128). 
        Each cell is a integer number representing quantized duration.
        """
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


def pr2grid(pr_mat, max_note_count=16, max_pitch=127, min_pitch=0,
                       pitch_pad_ind=130, dur_pad_ind=2,
                       pitch_sos_ind=128, pitch_eos_ind=129):
    pr_mat3d = np.ones((len(pr_mat), max_note_count, 6), dtype=int) * dur_pad_ind
    pr_mat3d[:, :, 0] = pitch_pad_ind
    pr_mat3d[:, 0, 0] = pitch_sos_ind
    cur_idx = np.ones(len(pr_mat), dtype=int)
    for t, p in zip(*np.where(pr_mat != 0)):
        pr_mat3d[t, cur_idx[t], 0] = p - min_pitch
        binary = np.binary_repr(min(int(pr_mat[t, p]), 32) - 1, width=5)
        pr_mat3d[t, cur_idx[t], 1: 6] = \
            np.fromstring(' '.join(list(binary)), dtype=int, sep=' ')
        if cur_idx[t] == max_note_count-1:
            continue
        cur_idx[t] += 1
    #print(cur_idx)
    pr_mat3d[np.arange(0, len(pr_mat)), cur_idx, 0] = pitch_eos_ind
    return pr_mat3d




def matrix2midi_with_dynamics(pr_matrices, programs, init_tempo=120, time_start=0, ACC=16):
    """
    Reconstruct a multi-track midi from a 3D matrix of shape (Track. Time, 128, 3).
    """
    tracks = []
    for program in programs:
        track_recon = pyd.Instrument(program=int(program), is_drum=False, name=pyd.program_to_instrument_name(int(program)))
        tracks.append(track_recon)

    indices_track, indices_onset, indices_pitch = np.nonzero(pr_matrices[:, :, :, 0])
    alpha = 1 / (ACC // 4) * 60 / init_tempo #timetep between each quntization bin
    for idx in range(len(indices_track)):
        track_id = indices_track[idx]
        onset = indices_onset[idx]
        pitch = indices_pitch[idx]

        start = onset * alpha
        duration = pr_matrices[track_id, onset, pitch, 0] * alpha
        velocity = pr_matrices[track_id, onset, pitch, 1]
        #if (int(velocity) > 127) or (int(velocity) < 0):
        #    print(velocity)
        #if int(pitch) > 127 or (int(pitch) < 0):
        #    print(pitch)

        note_recon = pyd.Note(velocity=int(velocity), pitch=int(pitch), start=time_start + start, end=time_start + start + duration)
        tracks[track_id].notes.append(note_recon)

    for idx in range(len(pr_matrices)):
        cc = []
        control_matrix = pr_matrices[idx, :, :, 2]
        for t, n in zip(*np.nonzero(control_matrix >= 0)):
            start = alpha * t
            #if int(n) > 127 or (int(n) < 0):
            #    print(n)
            #if int(control_matrix[t, n]) > 127 or (int(control_matrix[t, n]) < 0):
            #    print(int(control_matrix[t, n]))
            cc.append(pyd.ControlChange(int(n), int(control_matrix[t, n]), start))
        tracks[idx].control_changes = cc
    
    midi_recon = pyd.PrettyMIDI(initial_tempo=init_tempo)
    midi_recon.instruments = tracks
    return midi_recon


def matrix2midi_drum(pr_matrices, programs, init_tempo=120, time_start=0, ACC=64):
        """
        Reconstruct a multi-track midi from a 3D matrix of shape (Track. Time, 128, 3).
        """
        tracks = []
        for pogram in range(len(programs)):
            track_recon = pyd.Instrument(program=int(pogram), is_drum=True, name='drums')
            tracks.append(track_recon)

        indices_track, indices_onset, indices_pitch = np.nonzero(pr_matrices[:, :, :, 0])
        alpha = 1 / (ACC // 4) * 60 / init_tempo #timetep between each quntization bin
        for idx in range(len(indices_track)):
            track_id = indices_track[idx]
            onset = indices_onset[idx]
            pitch = indices_pitch[idx]

            start = onset * alpha
            duration = pr_matrices[track_id, onset, pitch, 0] * alpha
            velocity = pr_matrices[track_id, onset, pitch, 1]

            note_recon = pyd.Note(velocity=int(velocity), pitch=int(pitch), start=time_start + start, end=time_start + start + duration)
            tracks[track_id].notes.append(note_recon)

        for idx in range(len(pr_matrices)):
            cc = []
            control_matrix = pr_matrices[idx, :, :, 2]
            for t, n in zip(*np.nonzero(control_matrix > -1)):
                start = alpha * t
                cc.append(pyd.ControlChange(int(n), int(control_matrix[t, n]), start))
            tracks[idx].control_changes = cc
        
        #midi_recon = pyd.PrettyMIDI(initial_tempo=init_tempo)
        #midi_recon.instruments = tracks
        return tracks

def retrieve_control(pop909_dir, song, tracks):
    src_dir = os.path.join(pop909_dir.replace('quantization/POP09-PIANOROLL-4-bin-quantization/', 'POP909/'), song.split('.')[0], song.replace('.npz', '.mid'))
    src_midi = pyd.PrettyMIDI(src_dir)
    beats = src_midi.get_beats()
    beats = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))
    ACC = 4
    quantize = interp1d(np.array(range(0, len(beats))) * ACC, beats, kind='linear')
    quaver = quantize(np.array(range(0, (len(beats) - 1) * ACC)))

    pr_matrices, programs, _ = midi2matrix_with_dynamics(src_midi, quaver)

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
        #print(st.mode(diff).mode[0], st.mode(diff).count[0]/diff.shape[0])
    else:
        diff = from_midi[0][: mid_length] - from_4_bin[0][: mid_length]
        #print(st.mode(diff).mode[0], st.mode(diff).count[0]/diff.shape[0])
    return pr_matrices[:, :, :, 2: 3], st.mode(diff).mode[0]