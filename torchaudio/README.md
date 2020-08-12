# torchaudio
Basic operations in pytorch-based audio processing:

* Audio sequence conversion to mel-spectrogram

```
import torch
from scipy.io.wavfile import read
from mel import MelTransformer

sr, y = read('your-wavfile.wav')
mel_fn = MelTransformer(**your_kwargs)
mel = mel_fn.transform(torch.FloatTensor(y)[None])
```

* Griffin-Lim Algorithm

```
import torch
from scipy.io.wavfile import read
from mel import MelTransformer
from vocoders import griffin_lim

sr, y = read('your-wavfile.wav')
mel_fn = MelTransformer(**your_kwargs)
mel = mel_fn.transform(torch.FloatTensor(y)[None])[None]

mel_decompress = mel_transform.spectral_de_normalize(mel)
mel_decompress = mel_decompress.transpose(1, 2).data
spec_from_mel_scaling = 1000
spec_from_mel = torch.mm(mel_decompress[0], mel_transform.mel_basis)
spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
spec_from_mel = spec_from_mel * spec_from_mel_scaling

waveform = griffin_lim(spec_from_mel[:, :, :-1], mel_transform.stft_fn, 100)
```
