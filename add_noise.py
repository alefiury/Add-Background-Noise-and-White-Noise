import os
import glob
import random
import argparse
from typing import List, Tuple

import librosa
import numpy as np
from tqdm import tqdm
import soundfile as sf


def calculate_rms(samples: np.ndarray) -> float:
    """
    Given a numpy array of audio samples, return its Root Mean Square (RMS).

    Source: https://github.com/iver56/audiomentations

    ---

    Args:
        samples: Samples from a loaded audio file.

    Returns:
        Root Mean Square (RMS).
    """
    return np.sqrt(np.mean(np.square(samples)))


def calculate_desired_noise_rms(clean_rms: float,
                                snr: float) -> float:
    """
    Given the Root Mean Square (RMS) of a clean sound and a desired signal-to-noise ratio (SNR),
    calculate the desired RMS of a noise sound to be mixed in.

    Source: https://github.com/iver56/audiomentations
    Based on https://github.com/Sato-Kunihiko/audio-SNR/blob/8d2c933b6c0afe6f1203251f4877e7a1068a6130/create_mixed_audio_file.py#L20

    ---

    Args:
        clean_rms: Root Mean Square (RMS) - a value between 0.0 and 1.0
        snr: Signal-to-Noise (SNR) Ratio in dB - typically somewhere between -20 and 60

    Returns:
        Desired Root Mean Square (RMS).
    """

    a = float(snr) / 20
    noise_rms = clean_rms / (10 ** a)
    return noise_rms


def divide_dataset(audio_paths: List[str]) -> Tuple[List[str], List[str], List[str]]:
    """
    Divide dataset in 3 parts.

    ---
    Args:
        audio_paths: List of paths to the clean audio files.

    Returns:
        first_audio_set: First part of the dataset.
        second_audio_set: Second part of the dataset.
        third_audio_set: Third part of the dataset.
    """

    first_audio_set = audio_paths[:int(len(audio_paths)*(1/3))]
    second_audio_set = audio_paths[int(len(audio_paths)*(1/3)):int(len(audio_paths)*(2/3))]
    third_audio_set = audio_paths[int(len(audio_paths)*(2/3)):]

    return first_audio_set, second_audio_set, third_audio_set


def add_external_noise(audio_paths: List[str],
                        noise_paths: List[str],
                        sampling_rate: int,
                        SNR_LEVELS: List[int],
                        output_path: str) -> None:
    """
    Adds Background noise to a clean speech dataset based on pre-defined SNR levels.

    ---
    Args:
        audio_paths: List of paths to the clean audio files.
        noise_paths: List of paths to the noisy audio files.
        SNR_LEVELS: List of possible SNR levels.
        output_path: Base directory path to store the augmented audio files.

    Returns:
        None
    """

    for audio_path in tqdm(audio_paths):
        # Takes a random index to choose a random noisy audio file
        rand_index = random.randint(0, len(noise_paths)-1)
        noise = noise_paths[rand_index]
        # Choose randomly a SNR level
        SNR = random.choice(SNR_LEVELS)

        # Load noisy audio file
        sample_noise, _ = librosa.load(noise,
                            sr=sampling_rate,
                            mono=True,
                            res_type='kaiser_fast')

        # Load clean audio file
        sample_audio, _ = librosa.load(audio_path,
                                        sr=sampling_rate,
                                        mono=True,
                                        res_type='kaiser_fast')

        noise_rms = calculate_rms(sample_noise)
        clean_rms = calculate_rms(sample_audio)

        desired_noise_rms = calculate_desired_noise_rms(clean_rms, SNR)

        sample_noise = sample_noise * (desired_noise_rms / noise_rms)

        # Repeats noisy audio
        while sample_noise.shape[0] < sample_audio.shape[0]:
            sample_noise = np.concatenate((sample_noise, sample_noise))

        # Cut noisy audio
        if sample_noise.shape[0] > sample_audio.shape[0]:
            extra = sample_noise.shape[0]-sample_audio.shape[0]
            rand_start = random.randint(0, extra)
            rand_end = rand_start+sample_audio.shape[0]
            sample_noise = sample_noise[rand_start:rand_end]
            # sample_noise = sample_noise[:sample_audio.shape[0]]

        augmented_audio = sample_audio + sample_noise

        sf.write(os.path.join(output_path, os.path.basename(audio_path)), augmented_audio, sampling_rate, subtype='PCM_16')

def add_white_noise(audio_paths: List[str],
                        SNR_LEVELS: List[int],
                        sampling_rate: int,
                        output_path: str) -> None:
    """
    Adds Gaussian Noise to a clean speech dataset based on pre-defined SNR levels.

    ---
    Args:
        audio_paths: List of paths to the clean audio files.
        SNR_LEVELS: List of possible SNR levels.
        output_path: Base directory path to store the augmented audio files.

    Returns:
        None
    """

    for audio_path in tqdm(audio_paths):
        SNR = random.choice(SNR_LEVELS)

        sample_audio, _ = librosa.load(audio_path,
                                        sr=sampling_rate,
                                        mono=True,
                                        res_type='kaiser_fast')

        clean_rms = calculate_rms(sample_audio)

        desired_noise_rms = calculate_desired_noise_rms(clean_rms, SNR)

        white_noise = np.random.normal(0.0, desired_noise_rms, size=sample_audio.shape[0])

        augmented_audio = sample_audio + white_noise

        sf.write(os.path.join(output_path, os.path.basename(audio_path)), augmented_audio, sampling_rate, subtype='PCM_16')


def add_noise(dataset_base_path: str,
                first_noise_base_path: str,
                second_noise_base_path: str,
                sampling_rate: int,
                output_path: str,
                seed: int) -> None:
    """
    Prepares data.

    ---

    Args:
        dataset_base_path: Path to clean audio base directory.
        first_noise_base_path: Path to one of the noisy audio base directory.
        second_noise_base_path: Path to one of the noisy audio base directory.
        output_path: Path to the base directory where the augmented audios will be saved.
        seed: Seed used to shuffle the list of clean audio paths.

    Returns:
        None

    """

    audio_paths = glob.glob(os.path.join(dataset_base_path, '**', '*.wav'), recursive=True)
    random.Random(seed).shuffle(audio_paths)
    first_noise_paths = glob.glob(os.path.join(first_noise_base_path, '**', '*.wav'), recursive=True)
    second_noise_paths = glob.glob(os.path.join(second_noise_base_path, '**', '*.wav'), recursive=True)
    SNR_LEVELS = [0, 5, 10, 15, 20]

    os.makedirs(output_path, exist_ok=True)

    first_audio_set, second_audio_set, third_audio_set = divide_dataset(audio_paths)

    add_external_noise(audio_paths=first_audio_set,
                        noise_paths=first_noise_paths,
                        SNR_LEVELS=SNR_LEVELS,
                        sampling_rate=sampling_rate,
                        output_path=output_path)

    add_external_noise(audio_paths=second_audio_set,
                        noise_paths=second_noise_paths,
                        SNR_LEVELS=SNR_LEVELS,
                        sampling_rate=sampling_rate,
                        output_path=output_path)

    add_white_noise(audio_paths=third_audio_set,
                        SNR_LEVELS=SNR_LEVELS,
                        sampling_rate=sampling_rate,
                        output_path=output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_base_path', type=str)
    parser.add_argument('--first_noise_base_path', type=str)
    parser.add_argument('--second_noise_base_path', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sampling_rate', type=int, default=16_000)
    parser.add_argument('--output_path', type=str)
    args = parser.parse_args()

    add_noise(dataset_base_path=args.dataset_base_path,
                first_noise_base_path=args.first_noise_base_path,
                second_noise_base_path=args.second_noise_base_path,
                sampling_rate=args.sampling_rate,
                output_path=args.output_path,
                seed=args.seed)


if __name__ == '__main__':
    main()