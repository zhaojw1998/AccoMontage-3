import os
import pretty_midi as pyd 
import numpy as np 
import torch
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
from tqdm import tqdm

from piano_arranger.acc_utils import split_phrases
import piano_arranger.format_converter as cvt
from piano_arranger.models import DisentangleVAE, PolyDisVAE
from piano_arranger.AccoMontage import find_by_length, dp_search, re_harmonization, get_texture_filter, ref_spotlight

from orchestrator import Slakh2100_Pop909_Dataset, collate_fn, compute_pr_feat, EMBED_PROGRAM_MAPPING, Prior
from orchestrator.autoencoder_dataset import SLAKH_CLASS_PROGRAMS
from orchestrator.utils import grid2pr, pr2grid, matrix2midi, midi2matrix

from orchestrator.prior_dataset import TOTAL_LEN_BIN, ABS_POS_BIN, REL_POS_BIN

SLAKH_CLASS_MAPPING = {v: k for k, v in EMBED_PROGRAM_MAPPING.items()}


def load_premise(DATA_FILE_ROOT, DEVICE, load_piano_arranger=True):
    if load_piano_arranger:
        print('Loading lead sheet to piano arrangement module (piano arranger). This may take 1 or 2 minutes ...')
        data = np.load(os.path.join(DATA_FILE_ROOT, 'phrase_data.npz'), allow_pickle=True)
        melody = data['melody']
        acc = data['acc']
        chord = data['chord']
        vel = data['velocity']
        cc = data['cc']
        acc_pool = {}
        for LEN in tqdm(range(2, 17)):
            (mel, acc_, chord_, vel_, cc_, song_reference) = find_by_length(melody, acc, chord, vel, cc, LEN)
            acc_pool[LEN] = (mel, acc_, chord_, vel_, cc_, song_reference)
        texture_filter = get_texture_filter(acc_pool)
        edge_weights=np.load(os.path.join(DATA_FILE_ROOT, 'edge_weights.npz'), allow_pickle=True)

        piano_arranger = PolyDisVAE(DEVICE, chd_size=256, voi_size=256, txt_size=256, num_channel=10)
        piano_arranger.load_state_dict(torch.load(os.path.join(DATA_FILE_ROOT, "params_chord_texture.pt")))
        piano_arranger.to(DEVICE)
    
    else:
        piano_arranger, acc_pool, texture_filter, edge_weights = None, None, None, None

    print('Loading piano to multi-track arrangement module (orchestrator). This may take 1 or 2 minutes ...')
    slakh_dir = os.path.join(DATA_FILE_ROOT, 'Slakh2100_inference_set')
    dataset = Slakh2100_Pop909_Dataset(slakh_dir=slakh_dir, pop909_dir=None, debug_mode=False, split='inference', mode='train')

    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda b:collate_fn(b, DEVICE))
    REF = []
    REF_PROG = []
    REF_MIX = []
    for (mix, prog, function, _, _, _) in loader:
        mix = mix[:, :32]
        prog = prog[0, :]
        mix = mix.detach().cpu().numpy()[0]
        mix = grid2pr(mix, max_note_count=32)[np.newaxis, :, :]
        ref_mix = torch.from_numpy(np.concatenate(compute_pr_feat(mix)[1:], axis=-1)).to(function.device)

        REF.extend([batch for batch in function])
        REF_PROG.extend([prog for _ in range(len(function))])
        REF_MIX.append(ref_mix)
    REF_MIX = torch.cat(REF_MIX, dim=0)

    print('Initializing prior model')
    prior_model_path = os.path.join(DATA_FILE_ROOT, 'params_prior.pt')
    QaA_model_path = os.path.join(DATA_FILE_ROOT, "params_autoencoder.pt")
    orchestrator = Prior.init_inference_model(prior_model_path, QaA_model_path, DEVICE=DEVICE)
    orchestrator.to(DEVICE)
    orchestrator.eval()

    print('Finished.')
    return piano_arranger, orchestrator, (acc_pool, edge_weights, texture_filter), (REF, REF_PROG, REF_MIX)


def read_lead_sheet(DEMO_ROOT, SONG_NAME, SEGMENTATION, NOTE_SHIFT, melody_track_ID=0, filename='lead sheet.mid'):
    melody_roll, chord_roll = cvt.leadsheet2matrix(os.path.join(DEMO_ROOT, SONG_NAME, filename), melody_track_ID)
    assert(len(melody_roll == len(chord_roll)))
    if NOTE_SHIFT != 0:
        melody_roll = melody_roll[int(NOTE_SHIFT*4):, :]
        chord_roll = chord_roll[int(NOTE_SHIFT*4):, :]
    if len(melody_roll) % 16 != 0:
        pad_len = (len(melody_roll)//16+1)*16-len(melody_roll)
        melody_roll = np.pad(melody_roll, ((0, pad_len), (0, 0)))
        melody_roll[-pad_len:, -1] = 1
        chord_roll = np.pad(chord_roll, ((0, pad_len), (0, 0)))
        chord_roll[-pad_len:, 0] = -1
        chord_roll[-pad_len:, -1] = -1

    CHORD_TABLE = np.stack([cvt.expand_chord(chord) for chord in chord_roll[::4]], axis=0)
    LEADSHEET = np.concatenate((melody_roll, chord_roll[:, 1: -1]), axis=-1)    #T*142, quantized at 16th
    query_phrases = split_phrases(SEGMENTATION) #[('A', 8, 0), ('A', 8, 8), ('B', 8, 16), ('B', 8, 24)]

    midi_len = len(LEADSHEET)//16
    anno_len = sum([item[1] for item in query_phrases])
    if  midi_len > anno_len:
        LEADSHEET = LEADSHEET[: anno_len*16]
        CHORD_TABLE = CHORD_TABLE[: anno_len*4]
        print(f'Mismatch warning: Detect {midi_len} bars in the lead sheet (MIDI) and {anno_len} bars in the provided phrase annotation. The lead sheet is truncated to {anno_len} bars.')
    elif midi_len < anno_len:
        pad_len = (anno_len - midi_len)*16
        LEADSHEET = np.pad(LEADSHEET, ((0, pad_len), (0, 0)))
        LEADSHEET[-pad_len:, 129] = 1
        CHORD_TABLE = np.pad(CHORD_TABLE, ((0, pad_len//4), (0, 0)))
        CHORD_TABLE[-pad_len//4:, 11] = -1
        CHORD_TABLE[-pad_len//4:, -1] = -1
        print(f'Mismatch warning: Detect {midi_len} bars in the lead sheet (MIDI) and {anno_len} bars in the provided phrase annotation. The lead sheet is padded to {anno_len} bars.')

    melody_queries = []
    for item in query_phrases:
        start_bar = item[-1]
        length = item[-2]
        segment = LEADSHEET[start_bar*16: (start_bar+length)*16]
        melody_queries.append(segment)  #melody queries: list of T16*142, segmented by phrases
    
    return (LEADSHEET, CHORD_TABLE, melody_queries, query_phrases)


def read_piano_reduction(DEMO_ROOT, SONG_NAME, NOTE_SHIFT, melody_track_ID=0):
    ACC = 4 #quantize at 1/16 beat
    path = os.path.join(DEMO_ROOT, SONG_NAME, 'arrangement_piano.mid')
    midi = pyd.PrettyMIDI(path)
    beats = midi.get_beats()
    beats = np.append(beats, beats[-1] + (beats[-1] - beats[-2]))
    quantize = interp1d(np.array(range(0, len(beats))) * ACC, beats, kind='linear')
    quaver = quantize(np.array(range(0, (len(beats) - 1) * ACC)))

    piano_reduction = []
    for idx, track in enumerate(midi.instruments):
        if (not track.is_drum) and (idx != melody_track_ID):
            pr_matrix = np.zeros((len(quaver), 128))
            for note in track.notes:
                note_start = np.argmin(np.abs(quaver - note.start))
                note_end =  np.argmin(np.abs(quaver - note.end))
                if note_end == note_start:
                    note_end = min(note_start + 1, len(quaver) - 1)
                pr_matrix[note_start, note.pitch] = note_end - note_start
            piano_reduction.append(pr_matrix)
    piano_reduction = np.sum(np.array(piano_reduction), axis=0)

    melody = np.zeros((len(quaver), 128))
    for note in midi.instruments[melody_track_ID].notes:
        note_start = np.argmin(np.abs(quaver - note.start))
        note_end =  np.argmin(np.abs(quaver - note.end))
        if note_end == note_start:
            note_end = min(note_start + 1, len(quaver) - 1)
        melody[note_start, note.pitch] = note_end - note_start
    melody = np.array(melody)
    

    if NOTE_SHIFT != 0:
        piano_reduction = piano_reduction[int(NOTE_SHIFT*4)+1:, :]
        melody = melody[int(NOTE_SHIFT*4)+1:, :]

    if len(piano_reduction) % 32 != 0:
        pad_len = (len(piano_reduction)//32+1)*32-len(piano_reduction)
        piano_reduction = np.pad(piano_reduction, ((0, pad_len), (0, 0)))
        melody = np.pad(melody, ((0, pad_len), (0, 0)))

    return melody, piano_reduction


def piano_arrangement(pianoRoll, chord_table, melody_queries, query_phrases, acc_pool, edge_weights, texture_filter, piano_arranger, PREFILTER, tempo=100):
    print('Phrasal Unit selection begins:\n\t', f'{len(query_phrases)} phrases in the lead sheet;\n\t', f'set note density filter: {PREFILTER}.')
    phrase_indice, chord_shift = dp_search( melody_queries, 
                                            query_phrases, 
                                            acc_pool, 
                                            edge_weights, 
                                            texture_filter, 
                                            filter_id=PREFILTER)
    path = phrase_indice[0]
    shift = chord_shift[0]
    print('Re-harmonization begins ...')
    midi_recon, acc = re_harmonization(pianoRoll, chord_table, query_phrases, path, shift, acc_pool, model=piano_arranger, get_est=True, tempo=tempo)
    acc = np.array([grid2pr(matrix) for matrix in acc])
    print('Piano accompaiment generated!')

    return midi_recon, acc


def prompt_sampling(acc_piano, REF, REF_PROG, REF_MIX, MUST_HAVE=[], MUSTNOT_HAVE=[], DEVICE='cuda:0'):
    ref_mix = torch.from_numpy(np.concatenate(compute_pr_feat(acc_piano[0:1])[1:], axis=-1)).to(DEVICE)
    sim_func = torch.nn.CosineSimilarity(dim=-1)
    distance = sim_func(ref_mix, REF_MIX)
    distance = distance + torch.normal(mean=torch.zeros(distance.shape), std=0.2*torch.ones(distance.shape)).to(distance.device)

    MUSTNOT_HAVE = [(EMBED_PROGRAM_MAPPING[item]) for item in MUSTNOT_HAVE]
    MUST_HAVE = [EMBED_PROGRAM_MAPPING[item] for item in MUST_HAVE]
    for i in range(len(REF_PROG)):
        ref_i = [item.item() for item in REF_PROG[i]]
        distance[i] += len(set(MUST_HAVE).intersection(set(ref_i)))
        distance[i] -= len(set(MUSTNOT_HAVE).intersection(set(ref_i)))

    sim_values, anchor_points = torch.sort(distance, descending=True)
    IDX = 0
    #sim_value = sim_values[IDX]
    anchor_point = anchor_points[IDX]
    function = REF[anchor_point]
    prog = REF_PROG[anchor_point]
    prog_class = [SLAKH_CLASS_MAPPING[item.item()] for item in prog.cpu().detach().numpy()]
    program_name = [SLAKH_CLASS_PROGRAMS[item] for item in prog_class]
    print(f'Prior model initialized with {len(program_name)} tracks:\n\t{program_name}')
    return prog, function

def orchestration(acc_piano, chord_track, prog, function, orchestrator, DEVICE='cuda:0', blur=.5, p=.1, t=4, tempo=100, num_sample=1):
    print('Orchestration begins ...')
    if chord_track is not None:
        if len(acc_piano) > len(chord_track):
            chord_track = np.pad(chord_track, ((0, 0), (len(acc_piano)-len(chord_track))))
        else:
            chord_track = chord_track[:len(acc_piano)]
        acc_piano = np.max(np.stack([acc_piano, chord_track], axis=0), axis=0)

    mix = torch.from_numpy(np.array([pr2grid(matrix, max_note_count=32) for matrix in acc_piano])).to(DEVICE)
    r_pos = np.round(np.arange(0, len(mix), 1) / (len(mix)-1) * len(REL_POS_BIN))
    total_len = np.argmin(np.abs(TOTAL_LEN_BIN - len(mix))).repeat(len(mix))
    a_pos = np.append(ABS_POS_BIN[0: min(ABS_POS_BIN[-1],len(mix))], [ABS_POS_BIN[-1]] * (len(mix)-ABS_POS_BIN[-1]))
    r_pos = torch.from_numpy(r_pos).long().to(DEVICE)
    a_pos = torch.from_numpy(a_pos).long().to(DEVICE)
    total_len = torch.from_numpy(total_len).long().to(DEVICE)

    if function is not None:
        function = function.repeat(num_sample, 1, 1, *[1]*len(function.shape))
    recon_pitch, recon_dur = orchestrator.inference(mix.repeat(num_sample, *[1]*len(mix.shape)), prog.repeat(num_sample, *[1]*len(prog.shape)), function, total_len.unsqueeze(0), a_pos.unsqueeze(0), r_pos.unsqueeze(0), blur, p, t)  #function.unsqueeze(0).unsqueeze(0)

    grid_recon = torch.cat([recon_pitch.max(-1)[-1].unsqueeze(-1), recon_dur.max(-1)[-1]], dim=-1)
    batch, n_segments, track, _, max_simu_note, grid_dim = grid_recon.shape
    grid_recon = grid_recon.permute(0, 2, 1, 3, 4, 5)
    grid_recon = grid_recon.reshape(batch, track, -1, max_simu_note, grid_dim)

    midi_collection = []
    for batch_i in range(len(grid_recon)):
        pr_recon_ = np.array([grid2pr(matrix) for matrix in grid_recon[batch_i].detach().cpu().numpy()])
        pr_recon = matrix2midi(pr_recon_, [SLAKH_CLASS_MAPPING[item] for item in prog.cpu().detach().numpy()], tempo)
        midi_collection.append(pr_recon)
    print('Full-band accompaiment generated!')    
    return midi_collection
    