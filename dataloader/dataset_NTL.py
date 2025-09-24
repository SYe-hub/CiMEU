import os

from numpy.random import randint
from torch.utils import data

# from annotation.check import video_name
from dataloader.video_transform import *
import numpy as np
import torchaudio


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VideoDataset(data.Dataset):
    def __init__(self, dataset='DFEW', data_set=1, max_len=16, mode='train', data_mode='norm', is_face=True,
                 label_type='single', audio_conf=None):
        self.file_path = "./annotation/" + dataset + '/'
        self.dataset = dataset
        self.duration = 2
        self.num_segments = 8
        self.duration_audio = audio_conf.get('target_length') // self.num_segments
        self.dataset = dataset
        if dataset == 'MAFW':
            self.file_path += label_type + '/'
        if is_face:
            if data_mode == 'norm':
                self.data_path = '_face'
            elif data_mode == 'rv':
                self.data_path = '_face_rv'
            else:
                self.data_path = '_face_flow'
        else:
            if data_mode == 'norm':
                self.data_path = ''
            elif data_mode == 'rv':
                self.data_path = '_rv'
            else:
                self.data_path = '_flow'
        if dataset in ['DFEW', 'MAFW', 'RAVDESS', 'CREMA-D', 'eNTERFACE05', 'CASME2']:
            list_file = "set_" + str(data_set) + "_" + mode + ".txt"
        elif dataset == 'FERv39k':
            list_file = mode + "_All" + ".txt"
        else:
            list_file = mode + ".txt"
        file_name = ['th14_vit_g_16_4', 'th14_vit_g_16_8', 'th14_vit_g_16_16']
        # file_name = ['th14_vit_g_16_2', 'th14_vit_g_16_4', 'th14_vit_g_16_8']
        self.list_file = [self.file_path + file_name[0] + self.data_path + '/' + list_file,
                          self.file_path + file_name[1] + self.data_path + '/' + list_file,
                          self.file_path + file_name[2] + self.data_path + '/' + list_file]
        self.max_len = max_len
        self.mode = mode
        self.crop_ratio = [0.9, 1.0]
        self.input_noise = 0.0005
        self.num_frames_H = self.max_len
        self.num_frames_M = int(self.max_len / 2)
        self.num_frames_L = int(self.max_len / 4)
        self._parse_list()
        self.audio_conf = audio_conf
        self.melbins = self.audio_conf.get('num_mel_bins')
        self.freqm = self.audio_conf.get('freqm')
        self.timem = self.audio_conf.get('timem')
        print('now using following mask: {:d} freq, {:d} time'.format(self.audio_conf.get('freqm'),
                                                                      self.audio_conf.get('timem')))
        self.mixup = self.audio_conf.get('mixup')
        print('now using mix-up with rate {:f}'.format(self.mixup))
        self.dataset = self.audio_conf.get('dataset')
        print('now process ' + self.dataset)
        # dataset spectrogram mean and std, used to normalize the input
        self.norm_mean = self.audio_conf.get('mean')
        self.norm_std = self.audio_conf.get('std')
        # skip_norm is a flag that if you want to skip normalization to compute the normalization stats using src/get_norm_stats.py, if Ture, input normalization will be skipped for correctly calculating the stats.
        # set it as True ONLY when you are getting the normalization stats.
        self.skip_norm = self.audio_conf.get('skip_norm') if self.audio_conf.get('skip_norm') else False
        if self.skip_norm:
            print('now skip normalization (use it ONLY when you are computing the normalization stats).')
        else:
            print(
                'use dataset mean {:.3f} and std {:.3f} to normalize the input.'.format(self.norm_mean, self.norm_std))
        # if add noise for data augmentation
        self.noise = self.audio_conf.get('noise')
        if self.noise == True:
            print('now use noise augmentation')
        pass
    def _delete_data(self, mes_list):
        new_mes_list = []
        for mes in mes_list:
            path = mes[0]
            video_name = path.split('/')[-1].split('.')[0]
            if video_name in self.audio_list:
                new_mes_list.append(mes)
        return new_mes_list
    def _parse_list(self):
        # check the frame number is large >=16:
        # form is [video_id, num_frames, class_idx]
        audio_path = '/data3/LM/DFEW/Audio_1'
        audio_path = audio_path.replace('DFEW', self.dataset)
        audio_list = os.listdir(audio_path)
        self.audio_list = []
        for audio in audio_list:
            self.audio_list.append(audio.split('/')[-1].split('.')[0])
        tmp_H = [x.strip().split(' ') for x in open(self.list_file[0])]
        tmp_M = [x.strip().split(' ') for x in open(self.list_file[1])]
        tmp_L = [x.strip().split(' ') for x in open(self.list_file[2])]
        tmp_H = self._delete_data(tmp_H)
        tmp_M = self._delete_data(tmp_M)
        tmp_L = self._delete_data(tmp_L)
        self.video_list_H = [VideoRecord(item) for item in tmp_H]
        self.video_list_M = [VideoRecord(item) for item in tmp_M]
        self.video_list_L = [VideoRecord(item) for item in tmp_L]
        print(('video number:%d' % (len(self.video_list_H))))

    def _get_seq_frames(self, record, NUM_FRAMES):
        """
        Given the video index, return the list of sampled frame indexes.
        Args:
            index (int): the video index.
            temporal_sample_index (int): temporal sample index.
        Returns:
            seq (list): the indexes of frames of sampled from the video.
        """
        num_frames = NUM_FRAMES
        video_length = record.num_frames
        if video_length < num_frames:
            return [0]
        else:
            seg_size = float(video_length) / num_frames
            seq = []
            # index from 1, must add 1
            if self.mode == "train":
                for i in range(num_frames):
                    start = int(np.round(seg_size * i))
                    end = int(np.round(seg_size * (i + 1)))
                    seq.append(randint(start, end))
            else:
                duration = seg_size / 2  # 取中间那帧
                for i in range(num_frames):
                    start = int(np.round(seg_size * i))
                    # end = int(np.round(seg_size * (i + 1)))
                    frame_index = start + int(duration)
                    seq.append(frame_index)
            return seq
    def _get_train_audio_indices(self, n_frames):
        # split all frames into seg parts, then select frame in each part randomly
        average_duration = (n_frames - self.duration_audio + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                              size=self.num_segments)
        elif n_frames > self.num_segments:
            offsets = np.sort(randint(n_frames - self.duration_audio + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _get_test_audio_indices(self, n_frames):
        # split all frames into seg parts, then select frame in the mid of each part
        if n_frames > self.num_segments + self.duration_audio - 1:
            tick = (n_frames - self.duration_audio + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets

    def _wav2fbank(self, filename, filename2=None):
        # mixup
        if filename2 == None:
            waveform, sr = torchaudio.load(filename)
            waveform = waveform - waveform.mean()
        # mixup
        else:
            waveform1, sr = torchaudio.load(filename)
            waveform2, _ = torchaudio.load(filename2)

            waveform1 = waveform1 - waveform1.mean()
            waveform2 = waveform2 - waveform2.mean()

            if waveform1.shape[1] != waveform2.shape[1]:
                if waveform1.shape[1] > waveform2.shape[1]:
                    # padding
                    temp_wav = torch.zeros(1, waveform1.shape[1])
                    temp_wav[0, 0:waveform2.shape[1]] = waveform2
                    waveform2 = temp_wav
                else:
                    # cutting
                    waveform2 = waveform2[0, 0:waveform1.shape[1]]

            # sample lambda from uniform distribution
            #mix_lambda = random.random()
            # sample lambda from beta distribtion
            mix_lambda = np.random.beta(10, 10)

            mix_waveform = mix_lambda * waveform1 + (1 - mix_lambda) * waveform2
            waveform = mix_waveform - mix_waveform.mean()

        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
                                                  window_type='hanning', num_mel_bins=self.melbins, dither=0.0,
                                                  frame_shift=10)

        target_length = self.audio_conf.get('target_length')
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        # if p > 0:
        #     m = torch.nn.ZeroPad2d((0, 0, 0, p))
        #     fbank = m(fbank)
        # elif p < 0:
        #     # fbank = fbank[0:target_length, :]   # 默认从开始到固定的长度   考虑加入随机裁剪
        #     ran_frame = int(random.random() * abs(p))
        #     fbank = fbank[ran_frame: min(ran_frame + target_length, n_frames)]

        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
            n_frames = target_length
        if self.mode == 'train':
            segment_indices = self._get_train_audio_indices(n_frames)
        elif self.mode == 'test':
            segment_indices = self._get_test_audio_indices(n_frames)
        new_fbank = fbank[segment_indices[0]: segment_indices[0] + self.duration_audio, :]
        for seg_ind in segment_indices[1:]:
            new_fbank = torch.cat([new_fbank, fbank[seg_ind:seg_ind + self.duration_audio, :]], dim=0)
        fbank = new_fbank

        if filename2 == None:
            return fbank, 0
        else:
            return fbank, mix_lambda

    def _get_audio(self, record):
        video_name = record.path.split('/')[-1]
        audio_path = "/data3/LM/" + self.dataset + '/Audio_1' + '/' + video_name.replace('.npy', '.wav')
        fbank, mix_lambda = self._wav2fbank(audio_path)
        # SpecAug, not do for eval set
        if self.mode == 'train:':
            freqm = torchaudio.transforms.FrequencyMasking(self.freqm)
            timem = torchaudio.transforms.TimeMasking(self.timem)
            fbank = torch.transpose(fbank, 0, 1)
            # this is just to satisfy new torchaudio version, which only accept [1, freq, time]
            fbank = fbank.unsqueeze(0)
            if self.freqm != 0:
                fbank = freqm(fbank)
            if self.timem != 0:
                fbank = timem(fbank)
            # squeeze it back, it is just a trick to satisfy new torchaudio version
            fbank = fbank.squeeze(0)
            fbank = torch.transpose(fbank, 0, 1)

        # normalize the input for both training and test
        if not self.skip_norm:
            fbank = (fbank - self.norm_mean) / (self.norm_std * 2)
        # skip normalization the input if you are trying to get the normalization stats.
        else:
            pass

        if self.noise == True and self.mode != 'test':
            fbank = fbank + torch.rand(fbank.shape[0], fbank.shape[1]) * np.random.rand() / 10
            fbank = torch.roll(fbank, np.random.randint(-10, 10), 0)

        # mix_ratio = min(mix_lambda, 1 - mix_lambda) / max(mix_lambda, 1 - mix_lambda)

        # the output fbank shape is [time_frame_num, frequency_bins], e.g., [1024, 128]
        return fbank

    def __getitem__(self, index):
        record_H = self.video_list_H[index]
        record_M = self.video_list_M[index]
        record_L = self.video_list_L[index]
        # print(record_L.path)
        audio = self._get_audio(record_H)
        if self.mode == 'train':
            segment_indices_H = self._get_seq_frames(record_H, self.num_frames_H)
            segment_indices_M = self._get_seq_frames(record_M, self.num_frames_M)
            segment_indices_L = self._get_seq_frames(record_L, self.num_frames_L)
        elif self.mode == 'test':
            segment_indices_H = self._get_seq_frames(record_H, self.num_frames_H)
            segment_indices_M = self._get_seq_frames(record_M, self.num_frames_M)
            segment_indices_L = self._get_seq_frames(record_L, self.num_frames_L)
        feat, mask = self.get(record_H, segment_indices_H, 1)
        feat_M, mask_M = feat[:, ::2], mask[:, ::2]
        feat_L, mask_L = feat[:, ::4], mask[:, ::4]
        return (feat_L, mask_L), (feat_M, mask_M), (feat, mask), record_H.label, audio

    """按照特征长度来选择H，M，L"""
    def get(self, record, indices, n, padding_val=0.0):
        video_item = record.path
        video_item = "/data3/LM/" + video_item
        feats = np.load(video_item).astype(np.float32)
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))
        result = []
        if len(indices) == 1:
            result_feats = feats
        else:
            for seg_ind in indices:
                p = int(seg_ind)
                result.append(feats[:, p:p + 1])
            # 将result_feats中的tensor组成一个tensor
            result_feats = torch.cat(result, dim=1)
        cur_len = result_feats.shape[-1]
        batch_shape = [result_feats.shape[0], int(self.max_len / n)]
        batched_inputs = result_feats.new_full(batch_shape, padding_val)
        batched_inputs[:, :result_feats.shape[-1]].copy_(result_feats)
        if self.mode == 'train' and self.input_noise > 0:
            noise = torch.randn_like(batched_inputs) * self.input_noise
            batched_inputs += noise

        batched_masks = torch.arange(int(self.max_len / n))[None, :] < cur_len
        return batched_inputs, batched_masks

    def __len__(self):
        return len(self.video_list_H)


def train_data_loader(args, dataset, data_mode='norm', data_set=None, is_face=True, label_type='single'):
    train_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': args.freqm, 'timem': args.timem,
                  'mixup': args.mixup, 'dataset': args.dataset, 'mode': 'train', 'mean': args.dataset_mean,
                  'std': args.dataset_std,
                  'noise': args.noise}
    train_data = VideoDataset(dataset=dataset,
                              data_set=data_set,
                              max_len=16,
                              mode='train',
                              data_mode=data_mode,
                              is_face=is_face,
                              label_type=label_type,
                              audio_conf=train_audio_conf)
    return train_data


def test_data_loader(args, dataset, data_mode='norm', data_set=None, is_face=True, label_type='single'):
    val_audio_conf = {'num_mel_bins': 128, 'target_length': args.audio_length, 'freqm': 0, 'timem': 0, 'mixup': 0,
                      'dataset': args.dataset, 'mode': 'evaluation', 'mean': args.dataset_mean, 'std': args.dataset_std,
                      'noise': False}
    test_data = VideoDataset(dataset=dataset,
                             data_set=data_set,
                             max_len=16,
                             mode='test',
                             data_mode=data_mode,
                             is_face=is_face,
                             label_type=label_type,
                             audio_conf=val_audio_conf)
    return test_data
