import os
import pretty_midi as pyd 
import numpy as np 
import torch
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
from tqdm import tqdm

from piano_arranger.acc_utils import split_phrases
import piano_arranger.format_converter as cvt
from piano_arranger.models import DisentangleVAE
from piano_arranger.AccoMontage import find_by_length, dp_search, re_harmonization, get_texture_filter, ref_spotlight

from orchestrator import Slakh_Dataset, collate_fn, compute_pr_feat, EMBED_PROGRAM_MAPPING, Prior
from orchestrator.dataset import SLAKH_CLASS_PROGRAMS
from orchestrator.utils import grid2pr, pr2grid, matrix2midi, midi2matrix


SLAKH_CLASS_MAPPING = {v: k for k, v in EMBED_PROGRAM_MAPPING.items()}
TOTAL_LEN_BIN = np.array([4, 7, 12, 15, 20, 23, 28, 31, 36, 39, 44, 47, 52, 55, 60, 63, 68, 71, 76, 79, 84, 87, 92, 95, 100, 103, 108, 111, 116, 119, 124, 127, 132])
ABS_POS_BIN = np.arange(129)
REL_POS_BIN = np.arange(128)


def load_premise(DATA_FILE_ROOT, DEVICE):
    """Load AccoMontage Search Space"""
    print('Loading AccoMontage piano texture search space. This may take 1 or 2 minutes ...')
    data = np.load(os.path.join(DATA_FILE_ROOT, 'phrase_data.npz'), allow_pickle=True)
    melody = data['melody']
    acc = data['acc']
    chord = data['chord']
    vel = data['velocity']
    cc = data['cc']
    acc_pool = {}
    for LEN in tqdm(range(2, 11)):
        (mel, acc_, chord_, vel_, cc_, song_reference) = find_by_length(melody, acc, chord, vel, cc, LEN)
        acc_pool[LEN] = (mel, acc_, chord_, vel_, cc_, song_reference)
    texture_filter = get_texture_filter(acc_pool)
    edge_weights=np.load(os.path.join(DATA_FILE_ROOT, 'edge_weights.npz'), allow_pickle=True)

    """Load Q&A Prompt Search Space"""
    print('loading orchestration prompt search space ...')
    slakh_dir = os.path.join(DATA_FILE_ROOT, 'Slakh2100_inference_set')
    dataset = Slakh_Dataset(slakh_dir, debug_mode=False, split='test', mode='train')
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda b:collate_fn(b, DEVICE, get_pr_gt=True))
    REF_PR = []
    REF_P = []
    REF_T = []
    REF_PROG = []
    FLTN = []
    for (pr, _, prog, func_pitch, func_time, _, _, _, _, _) in tqdm(loader):
        pr = pr[0]
        prog = prog[0, :]
        func_pitch = func_pitch[0, :]
        func_time = func_time[0, :]
        fltn = torch.cat([torch.sum(func_pitch, dim=-2), torch.sum(func_time, dim=-2)], dim=-1)
        REF_PR.append(pr)
        REF_P.append(func_pitch[0])
        REF_T.append(func_time[0])
        REF_PROG.append(prog)
        FLTN.append(fltn[0])
    FLTN = torch.stack(FLTN, dim=0)

    """Initialize orchestration model (Prior + Q&A)"""
    print('Initialize model ...')
    prior_model_path = os.path.join(DATA_FILE_ROOT, 'params_prior.pt')
    QaA_model_path = os.path.join(DATA_FILE_ROOT, 'params_qa.pt')
    orchestrator = Prior.init_inference_model(prior_model_path, QaA_model_path, DEVICE=DEVICE)
    orchestrator.to(DEVICE)
    orchestrator.eval()
    piano_arranger = DisentangleVAE.init_model(torch.device('cuda')).cuda()
    piano_arranger.load_state_dict(torch.load(os.path.join(DATA_FILE_ROOT, 'params_reharmonizer.pt')))
    print('Finished.')
    return piano_arranger, orchestrator, (acc_pool, edge_weights, texture_filter), (REF_PR, REF_P, REF_T, REF_PROG, FLTN)


def read_lead_sheet(DEMO_ROOT, SONG_NAME, SEGMENTATION, NOTE_SHIFT, melody_track_ID=0):
    melody_roll, chord_roll = cvt.leadsheet2matrix(os.path.join(DEMO_ROOT, SONG_NAME, 'lead sheet.mid'), melody_track_ID)
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


def piano_arrangement(pianoRoll, chord_table, melody_queries, query_phrases, acc_pool, edge_weights, texture_filter, piano_arranger, PREFILTER):
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
    midi_recon, acc = re_harmonization(pianoRoll, chord_table, query_phrases, path, shift, acc_pool, model=piano_arranger, get_est=True, tempo=100)
    acc = np.array([grid2pr(matrix) for matrix in acc])
    print('Piano accompaiment generated!')
    return midi_recon, acc


def prompt_sampling(acc_piano, REF_PR, REF_P, REF_T, REF_PROG, FLTN, DEVICE='cuda:0'):
    fltn = torch.from_numpy(np.concatenate(compute_pr_feat(acc_piano[0:1]), axis=-1)).to(DEVICE)
    sim_func = torch.nn.CosineSimilarity(dim=-1)
    distance = sim_func(fltn, FLTN)
    distance = distance + torch.normal(mean=torch.zeros(distance.shape), std=0.2*torch.ones(distance.shape)).to(distance.device)
    sim_values, anchor_points = torch.sort(distance, descending=True)
    IDX = 0
    sim_value = sim_values[IDX]
    anchor_point = anchor_points[IDX]
    func_pitch = REF_P[anchor_point]
    func_time = REF_T[anchor_point]
    prog = REF_PROG[anchor_point]
    prog_class = [SLAKH_CLASS_MAPPING[item.item()] for item in prog.cpu().detach().numpy()]
    program_name = [SLAKH_CLASS_PROGRAMS[item] for item in prog_class]
    refr = REF_PR[anchor_point]
    midi_ref = matrix2midi(
        pr_matrices=refr.detach().cpu().numpy(), 
        programs=prog_class, 
        init_tempo=100)
    print(f'Prior model initialized with {len(program_name)} tracks:\n\t{program_name}')
    return midi_ref, (prog, func_pitch, func_time)

def orchestration(acc_piano, chord_track, prog, func_pitch, func_time, orchestrator, DEVICE='cuda:0', blur=.5, p=.1, t=4):
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

    if func_pitch is not None:
        func_pitch = func_pitch.unsqueeze(0).unsqueeze(0)
    if func_time is not None:
        func_time = func_time.unsqueeze(0).unsqueeze(0)
    recon_pitch, recon_dur = orchestrator.run_autoregressive_nucleus(mix.unsqueeze(0), prog.unsqueeze(0), func_pitch, func_time, total_len.unsqueeze(0), a_pos.unsqueeze(0), r_pos.unsqueeze(0), blur, p, t)  #func_pitch.unsqueeze(0).unsqueeze(0), func_time.unsqueeze(0).unsqueeze(0)

    grid_recon = torch.cat([recon_pitch.max(-1)[-1].unsqueeze(-1), recon_dur.max(-1)[-1]], dim=-1)
    bat_ch, track, _, max_simu_note, grid_dim = grid_recon.shape
    grid_recon = grid_recon.permute(1, 0, 2, 3, 4)
    grid_recon = grid_recon.reshape(track, -1, max_simu_note, grid_dim)

    pr_recon_ = np.array([grid2pr(matrix) for matrix in grid_recon.detach().cpu().numpy()])
    pr_recon = matrix2midi(pr_recon_, [SLAKH_CLASS_MAPPING[item.item()] for item in prog.cpu().detach().numpy()], 100)
    print('Full-band accompaiment generated!')
    return pr_recon
    