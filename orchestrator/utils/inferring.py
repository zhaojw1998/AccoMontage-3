import torch
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm


def mixture_function_prior(y_set_loader):
    collect = []
    print('Rendering sample space for style references ...')
    for (_, _, function_pitch, function_time), _, _ in tqdm(y_set_loader, total=len(y_set_loader)):
        collect.append(torch.cat([
                                torch.sum(function_pitch, dim=1).reshape(-1),
                                torch.sum(function_time, dim=1).reshape(-1)
                            ], dim=-1)
                    )
    return torch.stack(collect, dim=0)


def search_reference(x_fp, x_ft, y_mix_function_set):
    x_mix_function = torch.cat([
                            torch.sum(x_fp, dim=1).reshape(1, -1),
                            torch.sum(x_ft, dim=1).reshape(1, -1)
                        ], dim=-1)
    sim_func = torch.nn.CosineSimilarity(dim=-1)
    distance = sim_func(x_mix_function, y_mix_function_set)
    distance = distance + torch.normal(mean=torch.zeros(distance.shape), std=0.2*torch.ones(distance.shape)).to(distance.device)
    sim_value, anchor_point = distance.max(-1)
    return anchor_point

def velocity_adaption(velocity, x_tracks, y_mel=0):
    avg_velocity = np.mean(np.ma.masked_equal(velocity, value=0), axis=(1, 2))
    tgt_mel_vel_track = np.argmax(avg_velocity)
    if tgt_mel_vel_track != y_mel:
        tgt_mel_vel = velocity[tgt_mel_vel_track].copy()
        velocity[tgt_mel_vel_track] = velocity[y_mel]
        velocity[y_mel] = tgt_mel_vel
        avg_velocity = np.mean(np.ma.masked_equal(velocity, value=0), axis=(1, 2))

    new_velocity = np.zeros(velocity.shape)
    for idx_tk, track in enumerate(x_tracks):
        dyn_matrix =  velocity[idx_tk]
        masked_dyn_matrix = np.ma.masked_equal(dyn_matrix, value=0)
        mean = np.mean(masked_dyn_matrix, axis=-1)
        onsets = np.nonzero(mean.data)
        dynamic = mean.data[onsets]
        onsets = onsets[0].tolist()
        dynamic = dynamic.tolist()
        if len(dynamic) == 0:
            continue
        if not 0 in onsets:
            onsets = [0] + onsets
            dynamic = [dynamic[0]] + dynamic
        if not len(dyn_matrix)-1 in onsets:
            onsets = onsets + [len(dyn_matrix)-1]
            dynamic = dynamic + [dynamic[-1]]
        dyn_curve = interp1d(onsets, dynamic)
        for t, p in zip(*np.nonzero(track)):
            new_velocity[idx_tk, t, p] = dyn_curve(t)
    return new_velocity