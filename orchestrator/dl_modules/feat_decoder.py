import torch
from torch import nn
import random

class FeatDecoder(nn.Module):

    def __init__(self, z_input_dim=128,
                 hidden_dim=1024, z_dim=512, n_step=32, output_dim=3):
        super(FeatDecoder, self).__init__()
        self.z2dec_hid = nn.Linear(z_dim, hidden_dim)
        self.z2dec_in = nn.Linear(z_dim, z_input_dim)
        self.gru = nn.GRU(output_dim + z_input_dim, hidden_dim,
                          batch_first=True,
                          bidirectional=False)
        self.init_input = nn.Parameter(torch.rand(output_dim))
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim

        self.out = nn.Linear(hidden_dim, output_dim)

        self.sigmoid = nn.Sigmoid()
        self.output_dim = output_dim
        self.n_step = n_step
        self.bce_func = nn.BCELoss()
        self.mse_func = nn.MSELoss()

    def forward(self, z, inference, tfr, gt_feat=None):

        bs = z.size(0)

        z_hid = self.z2dec_hid(z).unsqueeze(0)

        z_in = self.z2dec_in(z).unsqueeze(1)

        if inference:
            tfr = 0.

        token = self.init_input.repeat(bs, 1).unsqueeze(1)

        out_feats = []

        for t in range(self.n_step):
            y_t, z_hid = \
                self.gru(torch.cat([token, z_in], dim=-1), z_hid)

            out_feat = self.out(y_t)  # (bs, 1, 3)

            onset_pred = self.sigmoid(out_feat[:, :, 0])
            rhy_int = out_feat[:, :, 1]
            pitch_center = out_feat[:, :, 2]

            out_feats.append(torch.stack([onset_pred, rhy_int, pitch_center], -1))

            # prepare the input to the next step
            if t == self.n_step - 1:
                break
            teacher_force = random.random() < tfr
            if teacher_force and not inference:
                token = gt_feat[:, t].unsqueeze(1)
            else:
                t_onset = onset_pred > 0.5
                t_int = rhy_int
                t_pitch = pitch_center
                token = torch.stack([t_onset, t_int, t_pitch], -1)

        recon = torch.cat(out_feats, dim=1)
        return recon

    def recon_loss(self, gt_feat, recon_feat):
        recon_onset = recon_feat[:, :, 0]
        recon_int = recon_feat[:, :, 1]
        recon_pitch = recon_feat[:, :, 2]

        onset_loss = self.bce_func(recon_onset, gt_feat[:, :, 0])
        int_loss = self.mse_func(recon_int, gt_feat[:, :, 1])
        pitch_loss = self.mse_func(recon_pitch, gt_feat[:, :, 2])

        loss = onset_loss + int_loss + pitch_loss

        return loss, onset_loss, int_loss, pitch_loss
