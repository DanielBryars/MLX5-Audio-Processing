# MLX5-Audio-Processing



- [UrbanSound Dataset](https://urbansounddataset.weebly.com/)
- [Freesound](https://freesound.org/)
- [Audacity on SourceForge](https://sourceforge.net/projects/audacity/)

### Data
- [Huggingface Dataset](https://huggingface.co/datasets/danavery/urbansound8K?library=datasets)


Note: couldn't get this to work
- [soundata] https://pypi.org/project/soundata/

git add . && git commit -m "WIP" && git push

## Scratch

https://www.kaggle.com/code/prabhavsingh/urbansound8k-classification



## Task 1
Mel-Frequency
Cepstral Coefficients (MFCC) from the audio slices using the
Essentia audio analysis library

In all experiments we extract the
features on a per-frame basis using a window size of 23.2 ms
and 50% frame overlap. We compute 40 Mel bands between
0 and 22050 Hz and keep the first 25 MFCC coefficients (we
do not apply any pre-emphasis nor liftering). The per-frame
values for each coefficient are summarized across time using
the following summary statistics: minimum, maximum, me-
dian, mean, variance, skewness, kurtosis and the mean and
variance of the first and second derivatives, resulting in a
feature vector of dimension 225 per slice.

