import numpy as np
from tensorboardX import SummaryWriter

from .plotting import plot_spectrogram_to_numpy


class MyWriter(SummaryWriter):
    def __init__(self, hp, logdir):
        super(MyWriter, self).__init__(logdir)
        self.hp = hp

    def log_training(self, train_loss, step):
        self.add_scalar('train_loss', train_loss, step)

    def log_evaluation(self, test_loss, sdr,
                       mixed_wav, target_wav, est_wav,
                       mixed_spec, target_spec, est_spec, est_mask,
                       step):
        
        self.add_scalar('test_loss', test_loss, step)
        self.add_scalar('SDR', sdr, step)

        self.add_audio('mixed_wav', mixed_wav, step, self.hp.audio.sample_rate)
        self.add_audio('target_wav', target_wav, step, self.hp.audio.sample_rate)
        self.add_audio('estimated_wav', est_wav, step, self.hp.audio.sample_rate)

        self.add_image('data/mixed_spectrogram',
            plot_spectrogram_to_numpy(mixed_spec), step, dataformats='HWC')
        self.add_image('data/target_spectrogram',
            plot_spectrogram_to_numpy(target_spec), step, dataformats='HWC')
        self.add_image('result/estimated_spectrogram',
            plot_spectrogram_to_numpy(est_spec), step, dataformats='HWC')
        self.add_image('result/estimated_mask',
            plot_spectrogram_to_numpy(est_mask), step, dataformats='HWC')
        self.add_image('result/estimation_error_sq',
            plot_spectrogram_to_numpy(np.square(est_spec - target_spec)), step, dataformats='HWC')
