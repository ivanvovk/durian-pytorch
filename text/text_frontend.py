import nltk
try:
    from listener import ipatoarpabet
except ImportError:
    raise ImportError("Please run `pip install git+https://github.com/shrubb/listener.git`.")


class TextFrontend(object):
    """
    Provides interface for conversion of symbolic phoneme text representation
    into a sequence of embedding ids for pushing them as inputs to TTS frontend.
    """
    PAD = '#'
    EOS = '~'
    PHONEME_CODES = 'AA1 AE0 AE1 AH0 AH1 AO0 AO1 AW0 AW1 AY0 AY1 B CH D DH EH0 EH1 EU0 EU1 EY0 EY1 F G HH IH0 IH1 IY0 IY1 JH K L M N NG OW0 OW1 OY0 OY1 P R S SH T TH UH0 UH1 UW0 UW1 V W Y Z ZH pau'.split()
    _PHONEME_SEP = ' '
    
    def __init__(self):
        self.SYMBOLS = [self.PAD, self.EOS] + self.PHONEME_CODES  # PAD should be first to have zero id
        self._symbol_to_id = {s: i for i, s in enumerate(self.SYMBOLS)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.SYMBOLS)}
        
    def _should_keep_token(self, token, token_dict):
        return token in token_dict \
            and token != self.PAD and token != self.EOS \
            and token != self._symbol_to_id[self.PAD] \
            and token != self._symbol_to_id[self.EOS]
    
    def forward(self, string):
        string = string.split(self._PHONEME_SEP) if isinstance(string, str) else string
        string.append(self.EOS)
        sequence = [self._symbol_to_id[s] for s in string
                    if self._should_keep_token(s, self._symbol_to_id)]
        return sequence
    
    def backward(self, sequence: list, use_eos=False):
        string = [self._id_to_symbol[idx] for idx in sequence
                  if self._should_keep_token(idx, self._id_to_symbol)]
        string = self._PHONEME_SEP.join(string)
        if use_eos:
            string = string.replace(self.EOS, '')
        return string
    
    def text_to_phonemes(self, text, custom_words={}):
        """
        Convert text into ARPAbet.
        For known words use CMUDict; for the rest try 'espeak' (to IPA) followed by 'listener'.
        :param text: str, input text.
        :param custom_words:
            dict {str: list of str}, optional
            Pronounciations (a list of ARPAbet phonemes) you'd like to override.
            Example: {'word': ['W', 'EU1', 'R', 'D']}
        :return: list of str, phonemes
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
