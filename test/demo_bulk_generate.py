import os
os.environ['CUDA_VISIBLE_DEVICES']= '0'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import pretty_midi as pyd 
import numpy as np 
import sys
sys.path.append('./')
from arrangement_utils import *
from piano_arranger.format_converter import matrix2leadsheet
import warnings
warnings.filterwarnings("ignore")


DEVICE = 'cuda:0'
TEMPO = 100
USE_PROMPT = False


DATA_FILE_ROOT = '/data1/zhaojw/data_file_dir/'
piano_arranger, orchestrator, piano_texture, band_prompt = load_premise(DATA_FILE_ROOT, DEVICE)


SONG_ROOT="test/test_samples"

for rnd in range(1, 4):
    for SONG_NAME in tqdm(os.listdir(SONG_ROOT)):
        with open(os.path.join(SONG_ROOT, SONG_NAME, 'begin_time.txt')) as f:
            NOTE_SHIFT = float(f.readlines()[0].replace('\n', ''))
        with open(os.path.join(SONG_ROOT, SONG_NAME, 'phrases.txt')) as f:
            SEGMENTATION = f.readlines()[0].replace('\n', '')   #Phrase labels precomputed by https://github.com/Dsqvival/hierarchical-structure-analysis

        lead_sheet = read_lead_sheet(SONG_ROOT, SONG_NAME, SEGMENTATION, NOTE_SHIFT, filename=SONG_NAME+'.mid')
        if len(lead_sheet[0]) > 32*16:
            continue #skip pieces longer than 32 bars for now
        save_name = f"test/demo_{rnd}/{SONG_NAME}"
        if not os.path.exists(save_name):
            os.makedirs(save_name)

        lead_sheet_recon = matrix2leadsheet(lead_sheet[0], tempo=100)
        lead_sheet_recon.write(f'{save_name}/lead_sheet.mid')

        #lead sheet to piano arrangement
        RD = np.random.randint(low=3, high=5)
        VN = np.random.randint(low=2, high=5)
        PREFILTER = (RD, VN)
        midi_piano, acc_piano = piano_arrangement(*lead_sheet, *piano_texture, piano_arranger, PREFILTER, TEMPO)
        #midi_piano.write(f'{save_name}/AccoMontage.mid')

        #piano to multi-track arrangement
        func_prompt = prompt_sampling(acc_piano, *band_prompt, DEVICE=DEVICE)
        instruments, _ = func_prompt
        midi_band = orchestration(acc_piano, None, instruments, None, orchestrator, DEVICE, blur=0.25, p=0.05, t=6, tempo=TEMPO)[0]
        mel_track = pyd.Instrument(program=1, is_drum=False, name='melody')
        mel_track.notes = midi_piano.instruments[0].notes
        midi_band.instruments.append(mel_track)
        midi_band.write(f'{save_name}/Ours.mid')


