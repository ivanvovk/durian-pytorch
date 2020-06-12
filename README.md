# DurIAN
Implementation of "Duration Informed Attention Network for Multimodal Synthesis" (https://arxiv.org/pdf/1909.01700.pdf) paper.

**Status**: released

# Info

DurIAN is encoder-decoder architecture for text-to-speech synthesis task. Unlike prior architectures like Tacotron 2 it doesn't learn explicit attention but takes into account phoneme durations information. So, to use this model one should have phonemized and duration-aligned dataset. However, you may try to use pretrained duration model on LJSpeech dataset (CMU dict used). Links will be provided below.

# Architecture details

DurIAN model consists of two modules: backbone synthesizer and duration predictor. However, current implementation contains baseline version and paper-based version.

## Baseline model

![pipeline](https://user-images.githubusercontent.com/9570420/81863803-6f0ba300-9574-11ea-9f02-481c2bba81f0.png)

Here are some of the most notable differences from vanilla DurIAN:
* Prosodic boundary markers aren't used (we didn't have them labeled), and thus there's no 'skip states' exclusion of prosodic boundaries' hidden states.
* Style codes aren't used too (same reason).
* Simpler network architectures.
* No Prenet in decoder.
* No attention used in decoder.
* Decoder's recurrent cell outputs single spectrogram frame at a time.
* Decoder's recurrent cell isn't conditioned on its own outputs (isn't "autogressive").

## Vanilla version

* Prosodic boundary markers aren't used (we didn't have them labeled), and thus there's no 'skip states' exclusion of prosodic boundaries' hidden states.
* Style codes aren't used too (same reason).
* Removed Prenet before CBHG encoder (didn't improved accuracy during experiments).

## Training

Both backbone synthesizer and duration models are trained simultaneously. For implementation simplifications duration model predicts alignment over the fixed max number of frames. You can learn this outputs as BCE problem, MSE problem by summing over frames-axis or to use both losses. Experiments showed that BCE-version of optimization process showed itself being unstable, so prefer using MSE (don't mind if you get bad alignments in Tensorboard)

# Reproducibility

You can check the synthesis demo wavfile (was obtained much before convergence) in `demo` folder (used Waveglow vocoder).

1. First of all, make sure you have installed all packages using `pip install --upgrade -r requirements.txt`. The code is test using `pytorch==1.5.0`.

2. Clone the repository: `git clone https://github.com/ivanvovk/DurrIAN`, then `cd DurIAN` and `git submodule update --init` to clone audio processing files.

3. To start training run `python train.py -c configs/default.json`. You can specify to train baseline model as `python train.py -c configs/baseline.json --baseline`

To make sure that everything works fine you may run unit tests in `tests` folder.

# Pretrained model and dataset aligning problem

The main drawback of this model is requiring of duration-aligned dataset. You can find parsed LJSpeech filelist used in the training of current implementation in `filelists` folder. In order to use your data, make sure you have organized your filelists in the same way as provided LJSpeech ones. Also, in order to save time and neurons of your brains you may try to train the model on your dataset without duration-aligning using the pretrained on LJSpeech duration model from my model checkpoint, which you may find via [this](https://drive.google.com/drive/folders/1eW9w7WHP2yp81-WafCpoOhvfDJSxckc_?usp=sharing) link.
