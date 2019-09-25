import os
import numpy as np
import matplotlib
import nlpaug.augmenter.audio as naa
import tensorflow as tf
from keras.layers import Add, Lambda, Concatenate, SpatialDropout1D

import keras
from keras.layers import Input, Activation, Dense, Conv1D, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.models import load_model, Model
from keras import optimizers
from keras.layers.advanced_activations import LeakyReLU
from  keras import backend as K
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import librosa.display
import librosa.filters
from joblib import Parallel, delayed
import multiprocessing
from constants import model_params, base_data_path
from scipy import signal
from scipy.io import wavfile
from skopt import gp_minimize
from skopt.space import Real
from functools import partial
from pydub import AudioSegment
from keras.utils import multi_gpu_model

from constants import *

#Set a random seed for numpy for reproducibility
np.random.seed(42)

if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    matplotlib.use('Agg')


try:
    import foundations
except Exception as e:
    print(e)



def load_wav(path, sr):
	return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	#proposed by @dsmiller
	wavfile.write(path, sr, wav.astype(np.int16))

def save_wavenet_wav(wav, path, sr, inv_preemphasize, k):
	# wav = inv_preemphasis(wav, k, inv_preemphasize)
	wav *= 32767 / max(0.01, np.max(np.abs(wav)))
	wavfile.write(path, sr, wav.astype(np.int16))

def preemphasis(wav, k, preemphasize=True):
	if preemphasize:
		return signal.lfilter([1, -k], [1], wav)
	return wav

def inv_preemphasis(wav, k, inv_preemphasize=True):
	if inv_preemphasize:
		return signal.lfilter([1], [1, -k], wav)
	return wav

#From https://github.com/r9y9/wavenet_vocoder/blob/master/audio.py
def start_and_end_indices(quantized, silence_threshold=2):
	for start in range(quantized.size):
		if abs(quantized[start] - 127) > silence_threshold:
			break
	for end in range(quantized.size - 1, 1, -1):
		if abs(quantized[end] - 127) > silence_threshold:
			break

	assert abs(quantized[start] - 127) > silence_threshold
	assert abs(quantized[end] - 127) > silence_threshold

	return start, end

def trim_silence(wav, hparams):
	'''Trim leading and trailing silence
	Useful for M-AILABS dataset if we choose to trim the extra 0.5 silence at beginning and end.
	'''
	#Thanks @begeekmyfriend and @lautjy for pointing out the params contradiction. These params are separate and tunable per dataset.
	return librosa.effects.trim(wav, top_db= hparams.trim_top_db, frame_length=hparams.trim_fft_size, hop_length=hparams.trim_hop_size)[0]

def get_hop_size(hparams):
	hop_size = hparams.hop_size
	if hop_size is None:
		assert hparams.frame_shift_ms is not None
		hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
	return hop_size

def linearspectrogram(wav, hparams):
	# D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
	D = _stft(wav, hparams)
	S = _amp_to_db(np.abs(D)**hparams.magnitude_power, hparams) - hparams.ref_level_db

	if hparams.signal_normalization:
		return _normalize(S, hparams)
	return S

def melspectrogram(wav, hparams):
	# D = _stft(preemphasis(wav, hparams.preemphasis, hparams.preemphasize), hparams)
	D = _stft(wav, hparams)
	S = _amp_to_db(_linear_to_mel(np.abs(D)**hparams.magnitude_power, hparams), hparams) - hparams.ref_level_db

	if hparams.signal_normalization:
		return _normalize(S, hparams)
	return S

def inv_linear_spectrogram(linear_spectrogram, hparams):
	'''Converts linear spectrogram to waveform using librosa'''
	if hparams.signal_normalization:
		D = _denormalize(linear_spectrogram, hparams)
	else:
		D = linear_spectrogram

	S = _db_to_amp(D + hparams.ref_level_db)**(1/hparams.magnitude_power) #Convert back to linear

	if hparams.use_lws:
		processor = _lws_processor(hparams)
		D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
		y = processor.istft(D).astype(np.float32)
		return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
	else:
		return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)


def inv_mel_spectrogram(mel_spectrogram, hparams):
	'''Converts mel spectrogram to waveform using librosa'''
	if hparams.signal_normalization:
		D = _denormalize(mel_spectrogram, hparams)
	else:
		D = mel_spectrogram

	S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db)**(1/hparams.magnitude_power), hparams)  # Convert back to linear

	if hparams.use_lws:
		processor = _lws_processor(hparams)
		D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
		y = processor.istft(D).astype(np.float32)
		return inv_preemphasis(y, hparams.preemphasis, hparams.preemphasize)
	else:
		return inv_preemphasis(_griffin_lim(S ** hparams.power, hparams), hparams.preemphasis, hparams.preemphasize)

###########################################################################################
# tensorflow Griffin-Lim
# Thanks to @begeekmyfriend: https://github.com/begeekmyfriend/Tacotron-2/blob/mandarin-new/datasets/audio.py

def inv_linear_spectrogram_tensorflow(spectrogram, hparams):
	'''Builds computational graph to convert spectrogram to waveform using TensorFlow.
	Unlike inv_spectrogram, this does NOT invert the preemphasis. The caller should call
	inv_preemphasis on the output after running the graph.
	'''
	if hparams.signal_normalization:
		D = _denormalize_tensorflow(spectrogram, hparams)
	else:
		D = linear_spectrogram

	S = tf.pow(_db_to_amp_tensorflow(D + hparams.ref_level_db), (1/hparams.magnitude_power))
	return _griffin_lim_tensorflow(tf.pow(S, hparams.power), hparams)

def inv_mel_spectrogram_tensorflow(mel_spectrogram, hparams):
	'''Builds computational graph to convert mel spectrogram to waveform using TensorFlow.
	Unlike inv_mel_spectrogram, this does NOT invert the preemphasis. The caller should call
	inv_preemphasis on the output after running the graph.
	'''
	if hparams.signal_normalization:
		D = _denormalize_tensorflow(mel_spectrogram, hparams)
	else:
		D = mel_spectrogram

	S = tf.pow(_db_to_amp_tensorflow(D + hparams.ref_level_db), (1/hparams.magnitude_power))
	S = _mel_to_linear_tensorflow(S, hparams)  # Convert back to linear
	return _griffin_lim_tensorflow(tf.pow(S, hparams.power), hparams)

###########################################################################################

def _lws_processor(hparams):
	import lws
	return lws.lws(hparams.n_fft, get_hop_size(hparams), fftsize=hparams.win_size, mode="speech")

def _griffin_lim(S, hparams):
	'''librosa implementation of Griffin-Lim
	Based on https://github.com/librosa/librosa/issues/434
	'''
	angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
	S_complex = np.abs(S).astype(np.complex)
	y = _istft(S_complex * angles, hparams)
	for i in range(hparams.griffin_lim_iters):
		angles = np.exp(1j * np.angle(_stft(y, hparams)))
		y = _istft(S_complex * angles, hparams)
	return y

def _griffin_lim_tensorflow(S, hparams):
	'''TensorFlow implementation of Griffin-Lim
	Based on https://github.com/Kyubyong/tensorflow-exercises/blob/master/Audio_Processing.ipynb
	'''
	with tf.variable_scope('griffinlim'):
		# TensorFlow's stft and istft operate on a batch of spectrograms; create batch of size 1
		S = tf.expand_dims(S, 0)
		S_complex = tf.identity(tf.cast(S, dtype=tf.complex64))
		y = tf.contrib.signal.inverse_stft(S_complex, hparams.win_size, get_hop_size(hparams), hparams.n_fft)
		for i in range(hparams.griffin_lim_iters):
			est = tf.contrib.signal.stft(y, hparams.win_size, get_hop_size(hparams), hparams.n_fft)
			angles = est / tf.cast(tf.maximum(1e-8, tf.abs(est)), tf.complex64)
			y = tf.contrib.signal.inverse_stft(S_complex * angles, hparams.win_size, get_hop_size(hparams), hparams.n_fft)
	return tf.squeeze(y, 0)

def _stft(y, hparams):
	if hparams.use_lws:
		return _lws_processor(hparams).stft(y).T
	else:
		return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=hparams.win_size, pad_mode='constant')

def _istft(y, hparams):
	return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams.win_size)

##########################################################
#Those are only correct when using lws!!! (This was messing with Wavenet quality for a long time!)
def num_frames(length, fsize, fshift):
	"""Compute number of time frames of spectrogram
	"""
	pad = (fsize - fshift)
	if length % fshift == 0:
		M = (length + pad * 2 - fsize) // fshift + 1
	else:
		M = (length + pad * 2 - fsize) // fshift + 2
	return M


def pad_lr(x, fsize, fshift):
	"""Compute left and right padding
	"""
	M = num_frames(len(x), fsize, fshift)
	pad = (fsize - fshift)
	T = len(x) + 2 * pad
	r = (M - 1) * fshift + fsize - T
	return pad, pad + r
##########################################################
#Librosa correct padding
def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
	'''compute right padding (final frame) or both sides padding (first and final frames)
	'''
	assert pad_sides in (1, 2)
	# return int(fsize // 2)
	pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
	if pad_sides == 1:
		return 0, pad
	else:
		return pad // 2, pad // 2 + pad % 2

# Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectogram, hparams):
	global _mel_basis
	if _mel_basis is None:
		_mel_basis = _build_mel_basis(hparams)
	return np.dot(_mel_basis, spectogram)

def _mel_to_linear(mel_spectrogram, hparams):
	global _inv_mel_basis
	if _inv_mel_basis is None:
		_inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
	return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _mel_to_linear_tensorflow(mel_spectrogram, hparams):
	global _inv_mel_basis
	if _inv_mel_basis is None:
		_inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
	return tf.transpose(tf.maximum(1e-10, tf.matmul(tf.cast(_inv_mel_basis, tf.float32), tf.transpose(mel_spectrogram, [1, 0]))), [1, 0])

def _build_mel_basis(hparams):
	assert hparams.fmax <= hparams.sample_rate // 2
	return librosa.filters.mel(hparams.sample_rate, hparams.n_fft, n_mels=hparams.num_mels,
							   fmin=hparams.fmin, fmax=hparams.fmax)

def _amp_to_db(x, hparams):
	min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
	return 20 * np.log10(np.maximum(min_level, x))

def _db_to_amp(x):
	return np.power(10.0, (x) * 0.05)

def _db_to_amp_tensorflow(x):
	return tf.pow(tf.ones(tf.shape(x)) * 10.0, x * 0.05)

def _normalize(S, hparams):
	if hparams.allow_clipping_in_normalization:
		if hparams.symmetric_mels:
			return np.clip((2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value,
			 -hparams.max_abs_value, hparams.max_abs_value)
		else:
			return np.clip(hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0, hparams.max_abs_value)

	assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0
	if hparams.symmetric_mels:
		return (2 * hparams.max_abs_value) * ((S - hparams.min_level_db) / (-hparams.min_level_db)) - hparams.max_abs_value
	else:
		return hparams.max_abs_value * ((S - hparams.min_level_db) / (-hparams.min_level_db))

def _denormalize(D, hparams):
	if hparams.allow_clipping_in_normalization:
		if hparams.symmetric_mels:
			return (((np.clip(D, -hparams.max_abs_value,
				hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value))
				+ hparams.min_level_db)
		else:
			return ((np.clip(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

	if hparams.symmetric_mels:
		return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
	else:
		return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

def _denormalize_tensorflow(D, hparams):
	if hparams.allow_clipping_in_normalization:
		if hparams.symmetric_mels:
			return (((tf.clip_by_value(D, -hparams.max_abs_value,
				hparams.max_abs_value) + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value))
				+ hparams.min_level_db)
		else:
			return ((tf.clip_by_value(D, 0, hparams.max_abs_value) * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)

	if hparams.symmetric_mels:
		return (((D + hparams.max_abs_value) * -hparams.min_level_db / (2 * hparams.max_abs_value)) + hparams.min_level_db)
	else:
		return ((D * -hparams.min_level_db / hparams.max_abs_value) + hparams.min_level_db)




# given a path, return list of all files in directory
def get_list_of_wav_files(file_path):
    files = os.listdir(file_path)
    absolute_given_dir = os.path.abspath(file_path)
    absolute_files = list(map(lambda file_path: os.path.join(absolute_given_dir, file_path), files))
    return absolute_files


def convert_to_flac(dir_path):
    for file_path in os.listdir(dir_path):
        if file_path.split('.')[-1] != "flac":
            read_file = AudioSegment.from_file(os.path.join(dir_path,file_path), file_path.split('.')[-1])
            os.remove(os.path.join(dir_path,file_path))
            base_name = file_path.split('.')[:-1]
            # read_file = read_file.set_channels(8)
            # base_name = ".".join(base_name)
            read_file.export(os.path.join(dir_path,f"{base_name[0]}.flac"), format="flac")

def get_target(file_path):
    if '/real/' in file_path:
        return 'real'
    elif '/fake/' in file_path:
        return 'fake'


def save_wav_to_npy(output_file, spectrogram):
    np.save(output_file, spectrogram)


def wav_to_mel(input_file, output_path):
    y, sr = librosa.load(input_file)
    filename = os.path.basename(input_file)
    target = get_target(input_file)

    output_file = '{}{}-{}'.format(output_path, filename.split('.')[0], target)

    mel_spectrogram_of_audio = librosa.feature.melspectrogram(y=y, sr=sr).T
    save_wav_to_npy(output_file, mel_spectrogram_of_audio)


def convert_and_save(real_audio_files, output_real, fake_audio_files, output_fake):
    for file in real_audio_files:
        wav_to_mel(file, output_real)
    print(str(len(real_audio_files)) + ' real files converted to spectrogram')

    for file in fake_audio_files:
        wav_to_mel(file, output_fake)
    print(str(len(fake_audio_files)) + ' fake files converted to spectrogram')


def split_title_line(title_text, max_words=5):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    seq = title_text.split()
    return '\n'.join([' '.join(seq[i:i + max_words]) for i in range(0, len(seq), max_words)])


def plot_spectrogram(pred_spectrogram, path, title=None, split_title=False, target_spectrogram=None, max_len=None,
                     auto_aspect=False):
    if max_len is not None:
        target_spectrogram = target_spectrogram[:max_len]
        pred_spectrogram = pred_spectrogram[:max_len]

    if split_title:
        title = split_title_line(title)

    fig = plt.figure(figsize=(10, 8))
    # Set common labels
    fig.text(0.5, 0.18, title, horizontalalignment='center', fontsize=16)

    # target spectrogram subplot
    if target_spectrogram is not None:
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)

        if auto_aspect:
            im = ax1.imshow(np.rot90(target_spectrogram), aspect='auto', interpolation='none')
        else:
            im = ax1.imshow(np.rot90(target_spectrogram), interpolation='none')
        ax1.set_title('Target Mel-Spectrogram')
        fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax1)
        ax2.set_title('Predicted Mel-Spectrogram')
    else:
        ax2 = fig.add_subplot(211)

    if auto_aspect:
        im = ax2.imshow(np.rot90(pred_spectrogram), aspect='auto', interpolation='none')
    else:
        im = ax2.imshow(np.rot90(pred_spectrogram), interpolation='none')
    fig.colorbar(mappable=im, shrink=0.65, orientation='horizontal', ax=ax2)

    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()


def process_audio_files(filename, dirpath):
    audio_array, sample_rate = librosa.load(os.path.join(dirpath, 'flac', filename), sr=16000)
    trim_audio_array, index = librosa.effects.trim(audio_array)
    mel_spec_array = melspectrogram(trim_audio_array, hparams=hparams).T
    # mel_spec_array = librosa.feature.melspectrogram(y=trim_audio_array, sr=sample_rate, n_mels=model_params['num_freq_bin']).T
    label_name = filename.split('_')[-1].split('.')[0]
    if (label_name == 'bonafide') or ('target' in label_name):
        label = 1
    elif label_name == 'spoof':
        label = 0
    else:
        label = None
    if label is None:
        print(f"Removing {filename} since it does not have label")
        os.remove(os.path.join(dirpath, 'flac', filename))
    return (mel_spec_array, label)


def convert_audio_to_processed_list(input_audio_array_list, filename, dirpath):
    label_name = filename.split('_')[-1].split('.')[0]
    out_list = []
    if (label_name == 'spoof'):
        audio_array_list = [input_audio_array_list[0]]
        choose_random_one_ind = np.random.choice(np.arange(1,len(input_audio_array_list)))
        audio_array_list.append(input_audio_array_list[choose_random_one_ind])
        label = 0
    elif (label_name == 'bonafide') or ('target' in label_name):
        audio_array_list = input_audio_array_list
        label = 1
    else:
        audio_array_list = [input_audio_array_list[0]]
        label = None

    for audio_array in audio_array_list:
        trim_audio_array, index = librosa.effects.trim(audio_array)
        mel_spec_array = melspectrogram(trim_audio_array, hparams=hparams).T
        # mel_spec_array = librosa.feature.melspectrogram(y=trim_audio_array, sr=sample_rate, n_mels=model_params['num_freq_bin']).T
        if label is None:
            print(f"Removing {filename} since it does not have label")
            os.remove(os.path.join(dirpath, 'flac', filename))
        out_list.append([mel_spec_array, label])
    return out_list



def process_audio_files_with_aug(filename, dirpath):
    sr = 16000
    audio_array, sample_rate = librosa.load(os.path.join(dirpath, 'flac', filename), sr=sr)
    aug_crop = naa.CropAug(sampling_rate=sr)
    audio_array_crop = aug_crop.augment(audio_array)
    aug_loud = naa.LoudnessAug(loudness_factor=(2, 5))
    audio_array_loud = aug_loud.augment(audio_array)
    aug_noise = naa.NoiseAug(noise_factor=0.03)
    audio_array_noise = aug_noise.augment(audio_array)

    audio_array_list= [audio_array,audio_array_crop,audio_array_loud,
                       audio_array_noise ]

    out_list = convert_audio_to_processed_list(audio_array_list, filename, dirpath)

    return out_list



def preprocess_and_save_audio_from_ray_parallel(dirpath, mode, recompute=False, dir_num=None, isaug=False):
    if isaug:
        preproc_filename = f'{mode}_preproc_aug.npy'
    else:
        preproc_filename = f'{mode}_preproc.npy'

    # if mode != 'train':
    #     preproc_filename = f'{mode}_preproc.npy'

    if dir_num is not None:
        base_path = base_data_path[dir_num]
    else:
        base_path = base_data_path[0]
    if not os.path.isfile(os.path.join(f'{base_path}/preprocessed_data', preproc_filename)) or recompute:
        filenames = os.listdir(os.path.join(dirpath, 'flac'))
        num_cores = multiprocessing.cpu_count()-1
        if isaug:
            precproc_list_saved = Parallel(n_jobs=num_cores)(
                delayed(process_audio_files_with_aug)(filename, dirpath) for filename in tqdm(filenames))
            # Flatten the list
            print(f"******original len of preproc_list: {len(precproc_list_saved)}")
            precproc_list = []
            for i in range(len(precproc_list_saved)):
                precproc_list.extend(precproc_list_saved[i])
            # precproc_list = [item for sublist in precproc_list for item in sublist]
            print(f"******flattened len of preproc_list: {len(precproc_list)}")
        else:
            precproc_list = Parallel(n_jobs=num_cores)(
                delayed(process_audio_files)(filename, dirpath) for filename in tqdm(filenames))

        precproc_list = [x for x in precproc_list if x[1] is not None]

        if not os.path.isdir(f'{base_path}/preprocessed_data'):
            os.mkdir(f'{base_path}/preprocessed_data')
        np.save(os.path.join(f'{base_path}/preprocessed_data', preproc_filename), precproc_list)
    else:
        print("Preprocessing already done!")


def process_audio_files_inference(filename, dirpath, mode):
   audio_array, sample_rate = librosa.load(os.path.join(dirpath, mode, filename), sr=16000)
   trim_audio_array, index = librosa.effects.trim(audio_array)
   mel_spec_array = melspectrogram(trim_audio_array, hparams=hparams).T
   if mode == 'unlabeled':
       return mel_spec_array
   elif mode == 'real':
       label = 1
   elif mode == 'fake':
       label = 0
   return mel_spec_array, label


def preprocess_from_ray_parallel_inference(dirpath, mode, use_parallel=True):
   filenames = os.listdir(os.path.join(dirpath, mode))
   if use_parallel:
       num_cores = multiprocessing.cpu_count()
       preproc_list = Parallel(n_jobs=num_cores)(
           delayed(process_audio_files_inference)(filename, dirpath, mode) for filename in tqdm(filenames))
   else:
       preproc_list=[]
       for filename in tqdm(filenames):
           preproc_list.append(process_audio_files_inference(filename, dirpath, mode))
   return preproc_list


def preprocess_and_save_audio_from_ray(dirpath, mode, recompute=False):
    filenames = os.listdir(os.path.join(dirpath, 'flac'))
    if not os.path.isfile(os.path.join(f'{base_data_path}/preprocessed_data', f'{mode}_preproc.npy')) or recompute:
        precproc_list = []
        for filename in tqdm(filenames):
            audio_array, sample_rate = librosa.load(os.path.join(dirpath, 'flac', filename), sr=16000)
            trim_audio_array, index = librosa.effects.trim(audio_array)
            mel_spec_array = melspectrogram(trim_audio_array, hparams=hparams).T
            # mel_spec_array = librosa.feature.melspectrogram(y=trim_audio_array, sr=sample_rate, n_mels=model_params['num_freq_bin']).T
            label_name = filename.split('_')[-1].split('.')[0]
            if label_name == 'bonafide':
                label = 1
            elif label_name == 'spoof':
                label = 0
            else:
                label = None
            if label is not None:
                precproc_list.append((mel_spec_array, label))
            if label is None:
                print("Removing {filename} since it does not have label")
                os.remove(os.path.join(dirpath, 'flac', filename))
        if not os.path.isdir(f'{base_data_path}/preprocessed_data'):
            os.mkdir(f'{base_data_path}/preprocessed_data')
        np.save(os.path.join(f'{base_data_path}/preprocessed_data', f'{mode}_preproc.npy'), precproc_list)
        # np.save(os.path.join(dirpath, 'preproc', 'preproc.npy'), precproc_list)
    else:
        print("Preprocessing already done!")


def preprocess_and_save_audio(dirpath, recompute=False):
    filenames = os.listdir(os.path.join(dirpath, 'flac'))
    if not os.path.isfile(os.path.join(dirpath, 'preproc', 'preproc.npy')) or recompute:
        precproc_list = []
        for filename in tqdm(filenames):
            audio_array, sample_rate = librosa.load(os.path.join(dirpath, 'flac', filename), sr=16000)
            trim_audio_array, index = librosa.effects.trim(audio_array)
            mel_spec_array = librosa.feature.melspectrogram(y=trim_audio_array, sr=sample_rate,
                                                            n_mels=model_params['num_freq_bin']).T
            label_name = filename.split('_')[-1].split('.')[0]
            if label_name == 'bonafide':
                label = 1
            elif label_name == 'spoof':
                label = 0
            else:
                label = None
            if label is not None:
                precproc_list.append((mel_spec_array, label))
            if label is None:
                print("Removing {filename} since it does not have label")
                os.remove(os.path.join(dirpath, 'flac', filename))
        if not os.path.isdir(os.path.join(dirpath, 'preproc')):
            os.mkdir(os.path.join(dirpath, 'preproc'))
        np.save(os.path.join(dirpath, 'preproc', 'preproc.npy'), precproc_list)
    else:
        print("Preprocessing already done!")


def describe_array(arr):
    print(f"Mean duration: {arr.mean()}\n Standard Deviation: {arr.std()}\nNumber of Clips: {len(arr)}")
    plt.hist(arr, bins=40)
    plt.show()


def get_durations_from_dir(audio_dir, file_extension='.wav'):
    durations = list()
    for root, dirs, filenames in os.walk(audio_dir):
        for file_name in filenames:
            if file_extension in file_name:
                file_path = os.path.join(root, file_name)
                audio = AudioSegment.from_wav(file_path)
                duration = audio.duration_seconds
                durations.append(duration)
    return np.array(durations)


def get_zero_pad(batch_input):
    # find max length
    max_length = np.max([len(x) for x in batch_input])
    for i, arr in enumerate(batch_input):
        curr_length = len(arr)
        pad_length = max_length - curr_length
        if len(arr.shape) > 1:
            arr = np.concatenate([arr, np.zeros((pad_length, arr.shape[-1]))])
        else:
            arr = np.concatenate([arr, np.zeros((pad_length))])
        batch_input[i] = arr
    return batch_input


def truncate_array(batch_input):
    min_arr_len = np.min([len(x) for x in batch_input])
    for i, arr in enumerate(batch_input):
        batch_input[i] = arr[:min_arr_len]
    return batch_input

def random_truncate_array(batch_input):
    min_arr_len = np.min([len(x) for x in batch_input])
    for i, arr in enumerate(batch_input):
        upper_limit_start_point = len(arr)-min_arr_len
        if upper_limit_start_point>0:
            start_point = np.random.randint(0,upper_limit_start_point)
        else:
            start_point = 0
        batch_input[i] = arr[start_point:(start_point+min_arr_len)]
    return batch_input


class f1_score_callback(keras.callbacks.Callback):
    def __init__(self, x_val_inp, y_val_inp, model_save_filename=None, save_model=True):
        self.x_val = x_val_inp
        self.y_val = y_val_inp
        self.model_save_filename = model_save_filename
        self.save_model = save_model
        self._val_f1 = 0

    def on_train_begin(self, logs={}):
        self.f1_score_value = []

    def on_epoch_end(self, epoch, logs={}):
        y_val = self.y_val
        datagen_val = DataGenerator(self.x_val, mode='test')
        y_pred = self.model.predict_generator(datagen_val, use_multiprocessing=False, max_queue_size=50)
        y_pred_labels = np.zeros((len(y_pred)))
        y_pred_labels[y_pred.flatten() > 0.5] = 1

        self._val_f1 = f1_score(y_val, y_pred_labels.astype(int))
        print(f"val_f1: {self._val_f1:.4f}")

        self.f1_score_value.append(self._val_f1)
        if self.save_model:
            if self._val_f1 >= max(self.f1_score_value):
                print("F1 score has improved. Saving model.")
                self.model.save(self.model_save_filename)

        try:
            foundations.log_metric('epoch_val_f1_score',self._val_f1)
            foundations.log_metric('best_f1_score', max(self.f1_score_value))
        except Exception as e:
            print(e)


        return


class DataGenerator(keras.utils.Sequence):
    def __init__(self, x_set, y_set=None, sample_weights=None, batch_size=model_params['batch_size'], shuffle=False, mode='train'):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.mode = mode
        self.sample_weights = sample_weights
        if self.mode != 'train':
            self.shuffle = False
        self.n = 0
        self.max = self.__len__()

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = get_zero_pad(batch_x)
        # batch_x = random_truncate_array(batch_x)
        batch_x = np.array(batch_x)
        batch_x = batch_x.reshape((len(batch_x), -1, hparams.num_mels))
        if self.mode != 'test':
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        # read your data here using the batch lists, batch_x and batch_y
        if self.mode == 'train':
            return np.array(batch_x), np.array(batch_y)
        if self.mode == 'val':
            return np.array(batch_x), np.array(batch_y)
        if self.mode == 'test':
            return np.array(batch_x)

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result


def customPooling(x):
    target = x[1]
    inputs = x[0]
    maskVal = 0
    #getting the mask by observing the model's inputs
    mask = K.equal(inputs, maskVal)
    mask = K.all(mask, axis=-1, keepdims=True)

    #inverting the mask for getting the valid steps for each sample
    mask = 1 - K.cast(mask, K.floatx())

    #summing the valid steps for each sample
    stepsPerSample = K.sum(mask, axis=1, keepdims=False)

    #applying the mask to the target (to make sure you are summing zeros below)
    target = target * mask

    #calculating the mean of the steps (using our sum of valid steps as averager)
    means = K.sum(target, axis=1, keepdims=False) / stepsPerSample

    return means



def build_custom_convnet():
    K.clear_session()
    image_input = Input(shape=(None, model_params['num_freq_bin']), name='image_input')

    num_conv_blocks = model_params['num_conv_blocks']
    init_neurons = model_params['num_conv_filters']
    spatial_dropout_fraction = model_params['spatial_dropout_fraction']
    num_dense_layers = model_params['num_dense_layers']
    num_dense_neurons = model_params['num_dense_neurons']
    learning_rate = model_params['learning_rate']
    convnet = []
    convnet_5 = []
    convnet_7 = []
    for ly in range(0, num_conv_blocks):
        if ly == 0:
            convnet.append(Conv1D(init_neurons, 3, strides=1, activation='linear', padding='causal')(image_input))
            convnet_5.append(Conv1D(init_neurons, 5, strides=1, activation='linear', padding='causal')(image_input))
            convnet_7.append(Conv1D(init_neurons, 7, strides=1, activation='linear', padding='causal')(image_input))
        else:
            convnet.append(
                Conv1D(init_neurons * (ly * 2), 3, strides=1, activation='linear', padding='causal')(convnet[ly - 1]))
            convnet_5.append(
                Conv1D(init_neurons * (ly * 2), 5, strides=1, activation='linear', padding='causal')(convnet_5[ly - 1]))
            convnet_7.append(
                Conv1D(init_neurons * (ly * 2), 7, strides=1, activation='linear', padding='causal')(convnet_7[ly - 1]))

        convnet[ly] = LeakyReLU()(convnet[ly])
        convnet_5[ly] = LeakyReLU()(convnet_5[ly])
        convnet_7[ly] = LeakyReLU()(convnet_7[ly])
        if model_params['residual_con'] > 0 and (ly - model_params['residual_con']) >= 0:
            res_conv = Conv1D(init_neurons * (ly * 2), 1, strides=1, activation='linear', padding='same')(
                convnet[ly - model_params['residual_con']])
            convnet[ly] = Add(name=f'residual_con_3_{ly}')([convnet[ly], res_conv])
            res_conv_5 = Conv1D(init_neurons * (ly * 2), 1, strides=1, activation='linear', padding='same')(
                convnet_5[ly - model_params['residual_con']])
            convnet_5[ly] = Add(name=f'residual_con_5_{ly}')([convnet_5[ly], res_conv_5])
            res_conv_7 = Conv1D(init_neurons * (ly * 2), 1, strides=1, activation='linear', padding='same')(
                convnet_7[ly - model_params['residual_con']])
            convnet_7[ly] = Add(name=f'residual_con_7_{ly}')([convnet_7[ly], res_conv_7])

        if ly<(num_conv_blocks-1):
            convnet[ly] = SpatialDropout1D(spatial_dropout_fraction)(convnet[ly])
            convnet_5[ly] = SpatialDropout1D(spatial_dropout_fraction)(convnet_5[ly])
            convnet_7[ly] = SpatialDropout1D(spatial_dropout_fraction)(convnet_7[ly])

    dense = Lambda(lambda x: customPooling(x))([image_input,convnet[ly]])
    dense_5 = Lambda(lambda x: customPooling(x))([image_input,convnet_5[ly]])
    dense_7 = Lambda(lambda x: customPooling(x))([image_input,convnet_7[ly]])

    dense = Concatenate()([dense, dense_5, dense_7])

    for layers in range(num_dense_layers):
        dense = Dense(num_dense_neurons, activation='linear')(dense)
        dense = BatchNormalization()(dense)
        dense = LeakyReLU()(dense)
        dense = Dropout(model_params['dense_dropout'])(dense)
    output_layer = Dense(1)(dense)
    output_layer = Activation('sigmoid')(output_layer)
    model = Model(inputs=image_input, outputs=output_layer)
    opt = optimizers.Adam(lr=learning_rate)
    try:
        model = multi_gpu_model(model, gpus=4)
    except:
        pass
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model



class Discriminator_Model():
    def __init__(self, load_pretrained=False, saved_model_name=None, real_test_mode=False):

        if not os.path.exists(model_params['model_save_dir']):
            os.makedirs(model_params['model_save_dir'])

        if not load_pretrained:
            self.model = build_custom_convnet()
            self.model.summary()
        else:
            self.model = load_model(os.path.join(f"./{model_params['model_save_dir']}",saved_model_name), custom_objects={'customPooling': customPooling})
        self.model_name = f"saved_model_{'_'.join(str(v) for k,v in model_params.items())}.h5"
        self.real_test_model_name = f"real_test_saved_model_{'_'.join(str(v) for k,v in model_params.items())}.h5"
        self.model_save_filename = os.path.join(f"./{model_params['model_save_dir']}", self.model_name)
        self.real_test_model_save_filename = os.path.join(f"./{model_params['model_save_dir']}", self.real_test_model_name)
        if real_test_mode:
            if run_on_foundations:
                self.real_test_data_dir = "/data/inference_data/"
            else:
                self.real_test_data_dir = "../data/inference_data/"

            # preprocess the files
            self.real_test_processed_data_real = preprocess_from_ray_parallel_inference(self.real_test_data_dir, "real", use_parallel=True)
            self.real_test_processed_data_fake = preprocess_from_ray_parallel_inference(self.real_test_data_dir,
                                                                                        "fake",
                                                                                        use_parallel=True)

            self.real_test_processed_data = self.real_test_processed_data_real + self.real_test_processed_data_fake
            self.real_test_processed_data = sorted(self.real_test_processed_data, key=lambda x: len(x[0]))
            self.real_test_features = [x[0] for x in self.real_test_processed_data]
            self.real_test_labels = [x[1] for x in self.real_test_processed_data]
            print(f"Length of real_test_processed_data: {len(self.real_test_processed_data)}")

    def train(self, xtrain, ytrain, xval, yval):
        callbacks = []
        tb = TensorBoard(log_dir='tflogs', write_graph=True, write_grads=False)
        callbacks.append(tb)

        try:
            foundations.set_tensorboard_logdir('tflogs')
        except:
            print("foundations command not found")

        es = EarlyStopping(monitor='val_loss', mode='min', patience=5, min_delta=0.0001,
                           verbose=1)
        callbacks.append(tb)
        callbacks.append(es)

        rp = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=2,
                               verbose=1)
        callbacks.append(rp)

        f1_callback = f1_score_callback(xval, yval, model_save_filename=self.model_save_filename)
        callbacks.append(f1_callback)

        class_weights = {1: 5, 0: 1}

        train_generator = DataGenerator(xtrain, ytrain)
        validation_generator = DataGenerator(xval, yval)
        self.model.fit_generator(train_generator,
                                 steps_per_epoch = len(train_generator),
                                 epochs = model_params['epochs'],
                                 validation_data=validation_generator,
                                 callbacks = callbacks,
                                 shuffle = False,
                                  use_multiprocessing = True,
                                  verbose = 1,
                                 class_weight =class_weights)

        self.model = load_model(self.model_save_filename, custom_objects={'customPooling': customPooling})

        try:
            foundations.save_artifact(self.model_save_filename, key='trained_model.h5')
        except:
            print("foundations command not found")


    def inference_on_real_data(self, threshold=0.5):
        datagen_val = DataGenerator(self.real_test_features, mode='test', batch_size=1)
        y_pred = self.model.predict_generator(datagen_val, use_multiprocessing=False, max_queue_size=50)
        y_pred_labels = np.zeros((len(y_pred)))
        y_pred_labels[y_pred.flatten() > threshold] = 1
        acc_score = accuracy_score(self.real_test_labels, y_pred_labels)
        f1_score_val = f1_score(self.real_test_labels, y_pred_labels)
        return acc_score, f1_score_val


    def get_labels_from_prob(self, y, threshold=0.5):
        y_pred_labels = np.zeros((len(y)))
        y = np.array(y)
        if isinstance(threshold, list):
            y_pred_labels[y.flatten() > threshold[0]] = 1
        else:
            y_pred_labels[y.flatten() > threshold] = 1
        return y_pred_labels

    def get_f1score_for_optimization(self, threshold, y_true, y_pred, ismin=False):
        y_pred_labels = self.get_labels_from_prob(y_pred, threshold=threshold)
        if ismin:
            return - f1_score(y_true, y_pred_labels)
        else:
            return f1_score(y_true, y_pred_labels)

    def predict_labels(self, x, threshold=0.5, raw_prob=False, batch_size=model_params['batch_size']):
        test_generator = DataGenerator(x, mode='test', batch_size=batch_size)
        y_pred = self.model.predict_generator(test_generator, steps=len(test_generator), max_queue_size=10)
        print(y_pred)
        if raw_prob:
            return y_pred
        else:
            y_pred_labels = self.get_labels_from_prob(y_pred, threshold=threshold)
            return y_pred_labels

    def optimize_threshold(self, xtrain, ytrain, xval, yval):
        ytrain_pred = self.predict_labels(xtrain, raw_prob=True)
        yval_pred = self.predict_labels(xval, raw_prob=True)
        self.opt_threshold = 0.5
        ytrain_pred_labels = self.get_labels_from_prob(ytrain_pred, threshold=self.opt_threshold)
        yval_pred_labels = self.get_labels_from_prob(yval_pred, threshold=self.opt_threshold)
        train_f1_score = f1_score(ytrain_pred_labels, ytrain)
        val_f1_score = f1_score(yval_pred_labels, yval)
        print(f"train f1 score: {train_f1_score}, val f1 score: {val_f1_score}")

        f1_train_partial = partial(self.get_f1score_for_optimization, y_true=ytrain.copy(), y_pred=ytrain_pred.copy(), ismin=True)
        n_searches = 50
        dim_0 = Real(low=0.2, high=0.8, name='dim_0')
        dimensions = [dim_0]
        search_result = gp_minimize(func=f1_train_partial,
                                    dimensions=dimensions,
                                    acq_func='gp_hedge',  # Expected Improvement.
                                    n_calls=n_searches,
                                    # n_jobs=n_cpu,
                                    verbose=False)

        self.opt_threshold = search_result.x
        if isinstance(self.opt_threshold,list):
            self.opt_threshold = self.opt_threshold[0]
        self.optimum_threshold_filename = f"model_threshold_{'_'.join(str(v) for k, v in model_params.items())}.npy"
        np.save(os.path.join(f"{model_params['model_save_dir']}",self.optimum_threshold_filename), self.opt_threshold)
        train_f1_score = self.get_f1score_for_optimization(self.opt_threshold, y_true=ytrain, y_pred=ytrain_pred)
        val_f1_score = self.get_f1score_for_optimization(self.opt_threshold, y_true=yval, y_pred=yval_pred )
        print(f"optimized train f1 score: {train_f1_score}, optimized val f1 score: {val_f1_score}")



    def evaluate(self, xtrain, ytrain, xval, yval, num_examples=1):
        ytrain_pred = self.predict_labels(xtrain, raw_prob=True)
        yval_pred = self.predict_labels(xval, raw_prob=True)
        try:
            self.optimum_threshold_filename = f"model_threshold_{'_'.join(str(v) for k, v in model_params.items())}.npy"
            self.opt_threshold = np.load(os.path.join(f"{model_params['model_save_dir']}",self.optimum_threshold_filename)).item()
            print(f"loaded optimum threshold: {self.opt_threshold}")
        except:
            self.opt_threshold = 0.5


        ytrain_pred_labels = self.get_labels_from_prob(ytrain_pred, threshold=self.opt_threshold)
        yval_pred_labels = self.get_labels_from_prob(yval_pred, threshold=self.opt_threshold)

        train_accuracy = accuracy_score(ytrain, ytrain_pred_labels)
        val_accuracy = accuracy_score(yval, yval_pred_labels)

        train_f1_score = f1_score(ytrain, ytrain_pred_labels)
        val_f1_score = f1_score(yval, yval_pred_labels)
        print (f"train accuracy: {train_accuracy}, train_f1_score: {train_f1_score},"
               f"val accuracy: {val_accuracy}, val_f1_score: {val_f1_score} ")

        try:
            foundations.log_metric('train_accuracy',np.round(train_accuracy,2))
            foundations.log_metric('val_accuracy', np.round(val_accuracy,2))
            foundations.log_metric('train_f1_score', np.round(train_f1_score,2))
            foundations.log_metric('val_f1_score', np.round(val_f1_score,2))
            foundations.log_metric('optimum_threshold', np.round(self.opt_threshold,2))
        except Exception as e:
            print(e)

        # True Positive Example
        ind_tp = np.argwhere(np.equal((yval_pred_labels + yval).astype(int), 2)).reshape(-1, )

        # True Negative Example
        ind_tn = np.argwhere(np.equal((yval_pred_labels + yval).astype(int), 0)).reshape(-1, )

        # False Positive Example
        ind_fp =np.argwhere( np.greater(yval_pred_labels, yval)).reshape(-1, )

        # False Negative Example
        ind_fn = np.argwhere(np.greater(yval, yval_pred_labels)).reshape(-1, )


        path_to_save_spetrograms = './spectrograms'
        if not os.path.isdir(path_to_save_spetrograms):
            os.makedirs(path_to_save_spetrograms)
        specs_saved = os.listdir(path_to_save_spetrograms)
        if len(specs_saved)>0:
            for file_ in specs_saved:
                os.remove(os.path.join(path_to_save_spetrograms,file_))

        ind_random_tp = np.random.choice(ind_tp, num_examples).reshape(-1,)
        tp_x = [xtrain[i] for i in ind_random_tp]

        ind_random_tn = np.random.choice(ind_tn, num_examples).reshape(-1,)
        tn_x = [xtrain[i] for i in ind_random_tn]

        ind_random_fp = np.random.choice(ind_fp, num_examples).reshape(-1,)
        fp_x = [xtrain[i] for i in ind_random_fp]

        ind_random_fn = np.random.choice(ind_fn, num_examples).reshape(-1,)
        fn_x = [xtrain[i] for i in ind_random_fn]

        print("Plotting spectrograms to show what the hell the model has learned")
        for i in range(num_examples):
            plot_spectrogram(tp_x[i], path=os.path.join(path_to_save_spetrograms, f'true_positive_{i}.png'))
            plot_spectrogram(tn_x[i], path=os.path.join(path_to_save_spetrograms,f'true_negative_{i}.png'))
            plot_spectrogram(fp_x[i], path=os.path.join(path_to_save_spetrograms,f'false_positive_{i}.png'))
            plot_spectrogram(fn_x[i], path=os.path.join(path_to_save_spetrograms,f'fale_negative_{i}.png'))

        try:
            foundations.save_artifact(os.path.join(path_to_save_spetrograms, f'true_positive_{i}.png'), key='true_positive_example')
            foundations.save_artifact(os.path.join(path_to_save_spetrograms,f'true_negative_{i}.png'), key='true_negative_example')
            foundations.save_artifact(os.path.join(path_to_save_spetrograms,f'false_positive_{i}.png'), key='false_positive_example')
            foundations.save_artifact(os.path.join(path_to_save_spetrograms,f'fale_negative_{i}.png'), key='false_negative_example')

        except Exception as e:
            print(e)
