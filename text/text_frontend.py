class TextFrontend(object):
    """
    Provides interface for conversion of symbolic phoneme text representation
    into a sequence of embedding ids for pushing them as inputs to TTS frontend.
    """
    PAD = '#'
    EOS = '~'
    PHONEME_CODES = sorted(lj_speech.POSSIBLE_PHONEME_CODES)
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
