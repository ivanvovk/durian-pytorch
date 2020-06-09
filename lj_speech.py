"""
LJ Speech dataset (https://keithito.com/LJ-Speech-Dataset/),
aligned and with audio converted to mel spectrograms.

Useful constants:
    POSSIBLE_PHONEME_CODES:
        set of str
        All possible phoneme codes found in the dataset.
    MEL_SCALE_BINS:
        int
        Number of channels in mel spectrograms.
    SAMPLING_RATE:
        int
        Original audio sampling rate, Hz.
    SPECTROGRAM_{MIN,MAX,MEAN,STD}
        float
        Spectrogram statistics in the dataset.
"""
import itertools
import json
from pathlib import Path
import pickle
import sys
import subprocess

import IPython.display

import numpy as np
import torch
import torch.utils.data

MEL_SCALE_BINS = 80
STFT_HOP_LENGTH = 256
SAMPLING_RATE = 22050

SPECTROGRAMS_MEAN = -5.529334
SPECTROGRAMS_STD = 2.0827525
SPECTROGRAMS_MIN = -11.52
SPECTROGRAMS_MAX = 2.08

# Defined below
POSSIBLE_PHONEME_CODES = None

def msec_to_frame(msec):
    """
    Given a time point in milliseconds, return the index of the nearest mel spectrogram frame.

    msec:
        float

    return:
    frame:
        int
    """
    return round(msec / 1000 * SAMPLING_RATE / STFT_HOP_LENGTH)

class LJSpeech(torch.utils.data.Dataset):
    """
    The master dataset that holds all available data.

    Each sample is a dictionary `data_dict` with the following items:
    'text'
        str
        Utterance text.
    'phonemes_start'
        list of int
        For each phoneme: an index of its first spectrogram frame.
    'phonemes_duration'
        list of int
        For each phoneme: the number of frames that it occupies.
        It's guaranteed that the phonemes don't overlap and have no gaps
        in between.
    'phonemes_code'
        list of str
        ARPAbet phoneme codes obtained
        from http://www.speech.cs.cmu.edu/cgi-bin/cmudict#phones.
    'spectrogram'
        numpy.ndarray, shape == (`MEL_SCALE_BINS`, T)
        Mel spectrogram of the spoken utterance.
    """
    def __init__(self, dataset_root):
        """
        dataset_root:
            pathlib.Path
            Path to a directory with .json and .npy files.
        """
        self.dataset_root = dataset_root
        
        # Read and store metafiles for each utterance
        self.utterances = []
        for utterance_labeling_file in sorted(self.dataset_root.iterdir()):
            if utterance_labeling_file.suffix == '.json':
                with open(utterance_labeling_file, 'r', encoding='utf8') as file:
                    utterance_labeling = json.load(file)

                # Filter 'broken' files
                if 'utterance' in utterance_labeling:
                    # Simplify structure a bit
                    utterance_labeling['words'] = utterance_labeling['utterance']['words']
                    del utterance_labeling['utterance']

                    self.utterances.append(utterance_labeling)
        
        assert len(self.utterances) == 13071

    def __len__(self):
        return len(self.utterances)
    
    def __getitem__(self, idx):
        utterance_labeling = self.utterances[idx]
        
        phonemes = [word['phones'] for word in utterance_labeling['words']]
        phonemes = list(itertools.chain(*phonemes))
        
        phonemes_start     = [msec_to_frame(phoneme['onset']) for phoneme in phonemes]
        phonemes_durations = \
            [b - a for a, b in zip(phonemes_start[:-1], phonemes_start[1:])] + \
            [msec_to_frame(phonemes[-1]['duration'])]
        phonemes_codes     = [phoneme['phone'] for phoneme in phonemes]

        spectrogram = np.load(
            (self.dataset_root / utterance_labeling['ID']).with_suffix('.mel.npy'))
        assert spectrogram.shape[0] == MEL_SCALE_BINS
        
        data_dict = {
            'text':              utterance_labeling['text'],
            'phonemes_start':    phonemes_start,
            'phonemes_duration': phonemes_durations,
            'phonemes_code':     phonemes_codes,
            'spectrogram':       spectrogram
        }
        return data_dict

def get_dataset(dataset_root):
    """
    Load the dataset and split it for training and validation.
    Also, initialize `lj_speech.POSSIBLE_PHONEME_CODES`.

    dataset_root:
        str or pathlib.Path
        Path to a directory with .json and .npy files.

    return:
    train_dataset:
        torch.utils.data.Dataset
        A subset of a LJSpeech instance.
    val_dataset:
        torch.utils.data.Dataset
        A subset of a LJSpeech instance.
    """
    global POSSIBLE_PHONEME_CODES
    with open(dataset_root / 'all_possible_phoneme_codes.pkl', 'rb') as f:
        POSSIBLE_PHONEME_CODES = pickle.load(f)

    dataset = LJSpeech(dataset_root)

    NUM_VAL_SAMPLES = 100
    train_dataset = torch.utils.data.Subset(dataset, range(len(dataset))[NUM_VAL_SAMPLES:])
    val_dataset   = torch.utils.data.Subset(dataset, range(len(dataset))[:NUM_VAL_SAMPLES])

    return train_dataset, val_dataset

class Vocoder:
    """
    Mel spectrogram to audio converter based on WaveGlow (https://github.com/NVIDIA/waveglow).
    """
    def __init__(self):
        for module_path in './waveglow/', './waveglow/tacotron2':
            if module_path not in sys.path:
                sys.path.insert(0, module_path)

        # Disable deprecation warnings
        import warnings
        warnings.simplefilter('ignore')

        self.waveglow = torch.load('waveglow_256channels_ljs_v2.pt')['model']
        self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
        self.waveglow.cuda().eval()

        from denoiser import Denoiser
        self.denoiser = Denoiser(self.waveglow).cuda()

        # Re-enable warnings
        warnings.resetwarnings()

    def __call__(self, spectrogram):
        """
        Convert mel spectrogram to raw audio.

        spectrogram:
            torch.Tensor, shape == (`MEL_SCALE_BINS`, T)
            Mel spectrogram.

        return:
        audio:
            torch.Tensor, shape == (t), device == 'cpu', 
        """
        assert torch.is_tensor(spectrogram)
        if spectrogram.ndim == 2:
            spectrogram = spectrogram[None]

        with torch.no_grad():
            mel = spectrogram.cuda()
            audio = self.waveglow.infer(mel, sigma=0.75)
            audio = self.denoiser(audio, 0.04)[0]
            return audio.cpu()

def play_audio(audio):
    """
    Play raw audio in Jupyter notebook.

    audio:
        torch.Tensor or numpy.ndarray, shape == (1, t)
        Raw audio, e.g. from `Vocoder`.

    return:
    widget:
        IPython.display.Audio
        Jupyter notebook widget.
    """
    return IPython.display.Audio(audio, rate=SAMPLING_RATE, autoplay=True)

try:
    from listener import ipatoarpabet
except ImportError:
    raise ImportError("Please run `pip install git+https://github.com/shrubb/listener.git`.")

try:
    import nltk
except ImportError:
    raise ImportError("Please install NLTK, e.g. by `pip install nltk`.")

def text_to_phonemes(text, custom_words={}):
    """
    Convert text into ARPAbet.
    For known words use CMUDict; for the rest try 'espeak' (to IPA) followed by 'listener'.

    text:
        str
        Input text.
    custom_words:
        dict {str: list of str}, optional
        Pronounciations (a list of ARPAbet phonemes) you'd like to override.
        Example: {'word': ['W', 'EU1', 'R', 'D']}

    return:
    phonemes:
        list of str
    """
    def convert_phoneme_CMU(phoneme):
        REMAPPING = {
            'AA0': 'AA1',
            'AA2': 'AA1',
            'AE2': 'AE1',
            'AH2': 'AH1',
            'AO2': 'AO1',
            'AW2': 'AW1',
            'AY2': 'AY1',
            'EH2': 'EH1',
            'ER0': 'EH1',
            'ER1': 'EH1',
            'ER2': 'EH1',
            'EY2': 'EY1',
            'IH2': 'IH1',
            'IY2': 'IY1',
            'OW2': 'OW1',
            'OY2': 'OY1',
            'UH2': 'UH1',
            'UW2': 'UW1',
        }
        return REMAPPING.get(phoneme, phoneme)

    def convert_phoneme_listener(phoneme):
        VOWELS = ['A', 'E', 'I', 'O', 'U']
        if phoneme[0] in VOWELS:
            phoneme += '1'
        return convert_phoneme_CMU(phoneme)

    try:
        known_words = nltk.corpus.cmudict.dict()
    except LookupError:
        nltk.download('cmudict')
        known_words = nltk.corpus.cmudict.dict()

    for word, phonemes in custom_words.items():
        known_words[word.lower()] = [phonemes]

    words = nltk.tokenize.WordPunctTokenizer().tokenize(text.lower())

    phonemes = []
    PUNCTUATION = '!?.,-:;"\'()'
    for word in words:
        if all(c in PUNCTUATION for c in word):
            pronounciation = ['pau']
        elif word in known_words:
            pronounciation = known_words[word][0]
            pronounciation = list(map(convert_phoneme_CMU, pronounciation))
        else:
            pronounciation = ipatoarpabet.translate(word)[0].split()
            pronounciation = list(map(convert_phoneme_listener, pronounciation))

        phonemes += pronounciation
        
    return phonemes
