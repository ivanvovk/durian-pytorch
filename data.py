import torch

from scipy.io.wavfile import read

from text import TextFrontend
from torchaudio.mel import MelTransformer


def str_to_int_list(s):
    return list(map(int, s.split()))


class Dataset(torch.utils.data.Dataset):
    """
    Your dataset should be of the following structure: wavs and their text filelist with transcriptions.
    In filelist be sure that each line consists: text, phonemes start, phonemes duration, phonemes, wav filename
    separated with "|".
    """
    def __init__(self, config, training=True):
        super(Dataset, self).__init__()
        self.training = training
        filelist = config['train_filelist'] if self.training else config['valid_filelist']
        with open(filelist, 'r') as f:
            self._metadata = [line.replace('\n', '') for line in f.readlines()]
        self._load_mels_from_disk = config['load_mels_from_disk']
        if not self._load_mels_from_disk:
            self.mel_fn = MelTransformer(
                filter_length=config['filter_length'],
                hop_length=config['hop_length'],
                win_length=config['win_length'],
                n_mel_channels=config['n_mel_channels'],
                sampling_rate=config['sampling_rate'],
                mel_fmin=config['mel_fmin'],
                mel_fmax=config['mel_fmax'],
                dynamic_range_compression='nvidia'
            )
            self.sampling_rate = config['sampling_rate']

    def _get_mel(self, filename):
        if self._load_mels_from_disk:
            return torch.load(filename)
        sr, y = read(filename)
        assert sr == self.sampling_rate, \
            f"""SR of file `{filename}` ({sr}) doesn't match SR from config {self.sampling_rate}."""
        mel = self.mel_fn.transform(torch.FloatTensor(y.astype(float)).reshape(1, -1))
        return mel

    def __getitem__(self, index):
        item_meta = self._metadata[index]
        text, phonemes_start, phonemes_duration, phonemes_code, filename = item_meta.split('|')
        
        item = {
            'text': text,
            'phonemes_start': str_to_int_list(phonemes_start),
            'phonemes_duration': str_to_int_list(phonemes_duration),
            'phonemes_code': phonemes_code.split(),
            'mel': self._get_mel(filename)
        }
        return item

    def __len__(self):
        return len(self._metadata)


class BatchCollate(object):
    """
    Collates batch objects with padding, decreasing sort by input length, etc.
    """
    def __init__(self, config):
        self.n_mel_channels = config['n_mel_channels']
        self.text_frontend = TextFrontend()
    
    def __call__(self, batch):
        B = len(batch)
        
        # Converting all phoneme representations into embedding ids
        for i, x in enumerate(batch):
            batch[i]['phonemes_code'] = self.text_frontend.forward(x['phonemes_code'])
        
        # Sorting batch by length of inputs
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x['phonemes_code']) for x in batch]), dim=0, descending=True
        )
        max_input_len = input_lengths[0]
        
        # Get max mel length
        max_target_len = max([x['mel'].shape[1] for x in batch])
        
        # Getting durations targets as alignment map and padding sequences
        alignments_padded = torch.zeros(B, max_input_len, max_target_len, dtype=torch.float32)
        sequences_padded = torch.zeros(B, max_input_len, dtype=torch.long)
        mels_padded = torch.zeros(B, self.n_mel_channels, max_target_len, dtype=torch.float32)
        output_lengths = torch.zeros(B).long()
        for index, i in enumerate(ids_sorted_decreasing):
            x = batch[i]
            assert len(x['phonemes_start']) == len(x['phonemes_duration'])
            for symbol_idx, (start, dur) in enumerate(zip(x['phonemes_start'], x['phonemes_duration'])):
                if not start + dur > max_target_len:
                    alignments_padded[index, symbol_idx, start:start+dur] = torch.ones(dur, dtype=torch.float32)
                else:
                    break
            sequence = x['phonemes_code']
            sequences_padded[index, :len(sequence)] = torch.LongTensor(sequence)
            mel = torch.FloatTensor(x['mel'])
            mels_padded[index, :, :mel.shape[1]] = mel
            output_lengths[index] = mel.shape[1]
        
        outputs = {
            'sequences_padded': sequences_padded,
            'mels_padded': mels_padded,
            'alignments_padded': alignments_padded,
            'input_lengths': input_lengths,
            'output_lengths': output_lengths
        }
        return outputs
