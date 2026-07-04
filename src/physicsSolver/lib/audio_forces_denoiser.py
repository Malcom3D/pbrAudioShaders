# Copyright (C) 2025 Malcom3D <malcom3d.gpl@gmail.com>
#
# This file is part of pbrAudio.
#
# pbrAudio is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pbrAudio is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pbrAudio.  If not, see <https://www.gnu.org/licenses/>.
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
from scipy import signal, ndimage
from scipy.ndimage import gaussian_filter1d
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from ..lib.force_data import ForceDataSequence

@dataclass
class AudioForcesDenoiser:
    """
    Post-processing class for denoising synthesized audio forces.
    
    Uses force data correlation for intelligent noise reduction while preserving
    transient events and maintaining audio quality.
    """
    
    # DC Offset Removal parameters
    enable_dc_blocker: bool = False
    dc_blocker_alpha: float = 0.999  # DC blocker coefficient (higher = more aggressive)
    
    # Adaptive Noise Gate parameters
    enable_noise_gate: bool = False
    gate_threshold_db: float = -60.0  # Noise gate threshold in dB
    gate_attack_ms: float = 2.0       # Attack time in ms
    gate_release_ms: float = 50.0     # Release time in ms
    gate_hold_ms: float = 10.0        # Hold time in ms
    
    # Temporal Smoothing parameters
    enable_temporal_smoothing: bool = False
    temporal_smoothing_window: int = 5  # Window size for temporal smoothing (samples)
    
    # Spectral Noise Reduction parameters
    enable_spectral_noise_reduction: bool = False
    spectral_fft_size: int = 2048      # FFT size for spectral processing
    spectral_hop_size: int = 512       # Hop size for spectral processing
    spectral_noise_floor_db: float = -80.0  # Noise floor estimate in dB
    spectral_reduction_strength: float = 0.8  # Reduction strength (0-1)
    spectral_smoothing: float = 0.3    # Spectral smoothing factor
    
    # Envelope Shaping parameters
    enable_envelope_shaping: bool = False
    envelope_attack_ms: float = 1.0    # Attack time for envelope
    envelope_release_ms: float = 20.0  # Release time for envelope
    envelope_smoothing: float = 0.5    # Envelope smoothing factor
    
    # Gaussian Adaptive Smoothing parameters
    enable_gaussian_adaptive_smoothing: bool = False
    gaussian_sigma_min: float = 0.5    # Minimum Gaussian sigma
    gaussian_sigma_max: float = 3.0    # Maximum Gaussian sigma
    gaussian_force_threshold: float = 0.1  # Force threshold for adaptive smoothing
    
    def __post_init__(self):
        """Initialize internal state."""
        self._sample_rate = None
        self._force_data = None
        self._noise_profile = None
        
    def process(self, audio_tracks: Dict[str, np.ndarray], force_data_sequence: ForceDataSequence, sample_rate: int) -> Dict[str, np.ndarray]:
        """
        Apply denoising to audio tracks using force data correlation.
        
        Parameters:
        -----------
        audio_tracks : Dict[str, np.ndarray]
            Dictionary of audio tracks to denoise
        force_data_sequence : ForceDataSequence
            Force data for correlation-based denoising
        sample_rate : int
            Audio sample rate
            
        Returns:
        --------
        Dict[str, np.ndarray]
            Denoised audio tracks
        """
        self._sample_rate = sample_rate
        self._force_data = force_data_sequence
        
        denoised_tracks = {}
        
        for track_name, track_data in audio_tracks.items():
            if len(track_data) == 0:
                denoised_tracks[track_name] = track_data
                continue
                
            # Apply denoising pipeline
                processed = track_data.copy()
                print('copy', processed.shape)
            
            # Step 1: DC Offset Removal
            if self.enable_dc_blocker:
                processed = self._remove_dc_offset(processed)
                print('_remove_dc_offset', processed.shape)
            
            # Step 2: Adaptive Noise Gate with force correlation
            if self.enable_noise_gate:
                processed = self._adaptive_noise_gate(processed, track_name)
                print('_adaptive_noise_gate', processed.shape)
            
            # Step 3: Temporal Smoothing
            if self.enable_temporal_smoothing:
                processed = self._temporal_smoothing(processed)
                print('_temporal_smoothing', processed.shape)
            
            # Step 4:: Spectral Noise Reduction
            if self.enable_spectral_noise_reduction:
                processed = self._spectral_noise_reduction(processed, track_name)
                print('_spectral_noise_reduction', processed.shape)
            
            # Step 5: Envelope Shaping
            if self.enable_envelope_shaping:
                processed = self._envelope_shaping(processed, track_name)
                print('_envelope_shaping', processed.shape)
            
            # Step 6: Gaussian Adaptive Smoothing
            if self.enable_gaussian_adaptive_smoothing:
                processed = self._gaussian_adaptive_smoothing(processed, track_name)
                print('_gaussian_adaptive_smoothing', processed.shape)
            
            denoised_tracks[track_name] = processed
            
        return denoised_tracks
    
    def _remove_dc_offset(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove DC offset using a high-pass filter (DC blocker).
        
        Parameters:
        -----------
        audio : np.ndarray
            Input audio signal
            
        Returns:
        --------
        np.ndarray
            Audio with DC offset removed
        """
        # Simple DC blocker filter: y[n] = x[n] - x[n-1] + alpha * y[n-1]
        y = np.zeros_like(audio)
        alpha = self.dc_blocker_alpha
        
        for n in range(1, len(audio)):
            y[n] = audio[n] - audio[n-1] + alpha * y[n-1]
        
        return y
    
    def _adaptive_noise_gate(self, audio: np.ndarray, track_name: str) -> np.ndarray:
        """
        Apply adaptive noise gate with force data correlation.
        
        Parameters:
        -----------
        audio : np.ndarray
            Input audio signal
        track_name : str
            Name of the track (for force correlation)
            
        Returns:
        --------
        np.ndarray
            Gated audio signal
        """
        # Convert threshold from dB to linear
        threshold_linear = 10 ** (self.gate_threshold_db / 20)
        
        # Calculate envelope
        envelope = np.abs(audio)
        
        # Smooth envelope
        attack_samples = int(self.gate_attack_ms * self._sample_rate / 1000)
        release_samples = int(self.gate_release_ms * self._sample_rate / 1000)
        hold_samples = int(self.gate_hold_ms * self._sample_rate / 1000)
        
        smoothed_envelope = self._smooth_envelope(envelope, attack_samples, release_samples, hold_samples)

        # Get force correlation factor
        force_factor = self._get_force_correlation(track_name, len(audio))
        
        # Apply adaptive threshold based on force data
        adaptive_threshold = threshold_linear * (1.0 + force_factor * 0.5)
        
        # Create gate
        gate = (smoothed_envelope > adaptive_threshold).astype(float)
        
        # Smooth gate transitions
        gate = gaussian_filter1d(gate, sigma=2.0)
        
        return audio * gate
    
    def _temporal_smoothing(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply temporal smoothing to reduce high-frequency noise.
        
        Parameters:
        -----------
        audio : np.ndarray
            Input audio signal
            
        Returns:
        --------
        np.ndarray
            Temporally smoothed audio
        """
        window = self.temporal_smoothing_window
        
        if window < 2:
            return audio
        
        # Apply moving average filter
        kernel = np.ones(window) / window
        smoothed = np.convolve(audio, kernel, mode='same')
        
        # Preserve transients by blending
        transient_mask = np.abs(audio - smoothed) > 0.1 * np.max(np.abs(audio))
        result = np.where(transient_mask, audio, smoothed)
        
        return result
    
    def _spectral_noise_reduction(self, audio: np.ndarray, track_name: str) -> np.ndarray:
        """
        Apply spectral noise reduction using STFT.
        
        Parameters:
        -----------
        audio : np.ndarray
            Input audio signal
        track_name : str
            Name of the track
            
        Returns:
        --------
        np.ndarray
            Spectrally denoised audio
        """
        # STFT parameters
        n_fft = self.spectral_fft_size
        hop_length = self.spectral_hop_size
        win_length = n_fft
        
        # Create window
        window = np.hanning(win_length)
        
        # Pad audio
        if len(audio) < win_length:
            return audio
        
        # Compute STFT
        stft = self._stft(audio, n_fft, hop_length, window)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise floor
        noise_floor = self._estimate_noise_floor(magnitude, track_name)
        
        # Compute spectral subtraction
        reduction = self.spectral_reduction_strength
        magnitude_denoised = np.maximum(magnitude - reduction * noise_floor, magnitude * (1 - reduction) * 0.01)
        
        # Apply spectral smoothing
        if self.spectral_smoothing > 0:
            magnitude_denoised = gaussian_filter1d(magnitude_denoised, sigma=self.spectral_smoothing, axis=0)
        
        # Reconstruct signal
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        audio_denoised = self._istft(stft_denoised, hop_length, window, len(audio))
        
        return audio_denoised
    
    def _envelope_shaping(self, audio: np.ndarray, track_name: str) -> np.ndarray:
        """
        Apply envelope shaping based on force data.
        
        Parameters:
        -----------
        audio : np.ndarray
            Input audio signal
        track_name : str
            Name of the track
            
        Returns:
        --------
        np.ndarray
            Envelope-shaped audio
        """
        # Get force envelope
        force_envelope = self._get_force_envelope(track_name, len(audio))
        
        # Smooth force envelope
        attack_samples = int(self.envelope_attack_ms * self._sample_rate / 1000)
        release_samples = int(self.envelope_release_ms * self._sample_rate / 1000)
        
        smoothed_force = self._smooth_envelope(force_envelope, attack_samples, release_samples, 0)
        
        # Apply envelope smoothing
        if self.envelope_smoothing > 0:
            smoothed_force = gaussian_filter1d(smoothed_force, sigma=self.envelope_smoothing * 10)
        
        # Normalize force envelope
        if np.max(smoothed_force) > 0:
            smoothed_force = smoothed_force / np.max(smoothed_force)
        
        # Apply envelope to audio
        shaped_audio = audio * (0.5 + 0.5 * smoothed_force)
        
        return shaped_audio
    
    def _gaussian_adaptive_smoothing(self, audio: np.ndarray, track_name: str) -> np.ndarray:
        """
        Apply Gaussian adaptive smoothing based on force data.
        
        Parameters:
        -----------
        audio : np.ndarray
            Input audio signal
        track_name : str
            Name of the track
            
        Returns:
        --------
        np.ndarray
            Adaptively smoothed audio
        """
        # Get force magnitude
        force_magnitude = self._get_force_magnitude(track_name, len(audio))
        
        # Normalize force magnitude
        if np.max(force_magnitude) > 0:
            force_magnitude = force_magnitude / np.max(force_magnitude)
        
        # Calculate adaptive sigma
        sigma = self.gaussian_sigma_min + (self.gaussian_sigma_max - self.gaussian_sigma_min) * (1.0 - force_magnitude)
        
        # Apply adaptive Gaussian smoothing
        smoothed = np.zeros_like(audio)
        for i in range(len(audio)):
            # Use local sigma for each sample
            local_sigma = sigma[i]
            if local_sigma > 0:
                # Apply Gaussian filter to local window
                window_size = int(local_sigma * 3)
                start = max(0, i - window_size)
                end = min(len(audio), i + window_size + 1)
                
                weights = np.exp(-0.5 * ((np.arange(start, end) - i) / local_sigma) ** 2)
                weights = weights / np.sum(weights)
                
                smoothed[i] = np.sum(audio[start:end] * weights)
            else:
                smoothed[i] = audio[i]
        
        # Blend with original to preserve transients
        transient_mask = force_magnitude > self.gaussian_force_threshold
        result = np.where(transient_mask, audio, smoothed)
        
        return result
    
    def _get_force_correlation(self, track_name: str, length: int) -> np.ndarray:
        """
        Get force correlation factor for adaptive processing.
        
        Parameters:
        -----------
        track_name : str
            Name of the track
        length : int
            Length of the audio signal
            
        Returns:
        --------
        np.ndarray
            Force correlation factor (0-1)
        """
        if self._force_data is None:
            return np.zeros(length)
        
        # Map track name to force data
        force_map = {
            'impact': 'normal_force_magnitude',
            'sliding': 'tangential_force_magnitude',
            'scraping': 'tangential_force_magnitude',
            'rolling': 'normal_force_magnitude',
            'sliding_sound': 'tangential_force_magnitude',
            'scraping_sound': 'tangential_force_magnitude',
            'rolling_sound': 'normal_force_magnitude',
            'non_collision': 'normal_force_magnitude',
            'coupling_strength': 'coupling_strength'
        }
        
        force_key = force_map.get(track_name, 'normal_force_magnitude')
        
        # Get force data at regular intervals
        force_values = np.zeros(length)
        step = max(1, len(self._force_data.frames) // length)
        
        for i in range(length):
            frame_idx = min(i * step, len(self._force_data.frames) - 1)
            try:
                if force_key == 'normal_force_magnitude':
                    force_values[i] = self._force_data.get_normal_force_magnitude(self._force_data.frames[frame_idx])
                elif force_key == 'tangential_force_magnitude':
                    force_values[i] = self._force_data.get_tangential_force_magnitude(self._force_data.frames[frame_idx])
                elif force_key == 'coupling_strength':
                    force_values[i] = self._force_data.get_coupling_strength(self._force_data.frames[frame_idx])
            except:
                   force_values[i] = 0
        
        # Normalize
        if np.max(force_values) > 0:
            force_values = force_values / np.max(force_values)
        
        return force_values
    
    def _get_force_envelope(self, track_name: str, length: int) -> np.ndarray:
        """
        Get force envelope for envelope shaping.
        
        Parameters:
        -----------
        track_name : str
            Name of the track
        length : int
            Length of the audio signal
            
        Returns:
        --------
        np.ndarray
            Force envelope
        """
        return self._get_force_correlation(track_name, length)
    
    def _get_force_magnitude(self, track_name: str, length: int) -> np.ndarray:
        """
        Get force magnitude for adaptive smoothing.
        
        Parameters:
        -----------
        track_name : str
            Name of the track
        length : int
            Length of the audio signal
            
        Returns:
        --------
        np.ndarray
            Force magnitude
        """
        return self._get_force_correlation(track_name, length)
    
    def _estimate_noise_floor(self, magnitude: np.ndarray, track_name: str) -> np.ndarray:
        """
        Estimate noise floor from magnitude spectrum.
        
        Parameters:
        -----------
        magnitude : np.ndarray
            Magnitude spectrogram
        track_name : str
            Name of the track
            
        Returns:
        --------
        np.ndarray
            Estimated noise floor
        """
        # Use minimum statistics for noise estimation
        noise_floor = np.min(magnitude, axis=1, keepdims=True)
        
        # Apply smoothing
        noise_floor = gaussian_filter1d(noise_floor, sigma=2.0, axis=0)
        
        # Apply floor based on force data
        force_factor = np.mean(self._get_force_correlation(track_name, magnitude.shape[1]))
        
        # Scale noise floor based on force activity
        noise_floor = noise_floor * (1.0 + force_factor * 0.5)
        
        return noise_floor
    
    def _smooth_envelope(self, envelope: np.ndarray, attack_samples: int, release_samples: int, hold_samples: int) -> np.ndarray:
        """
        Smooth envelope with separate attack and release times.
        
        Parameters:
        -----------
        envelope : np.ndarray
            Input envelope
        attack_samples : int
            Attack time in samples
        release_samples : int
            Release time in samples
        hold_samples : int
            Hold time in samples
            
        Returns:
        --------
        np.ndarray
            Smoothed envelope
        """
        smoothed = np.zeros_like(envelope)
        
        # Apply attack/release smoothing
        for i in range(len(envelope)):
            if i == 0:
                smoothed[i] = envelope[i]
            else:
                if envelope[i] > smoothed[i-1]:
                    # Attack phase
                    alpha = 1.0 / max(attack_samples, 1)
                    smoothed[i] = (1 - alpha) * smoothed[i-1] + alpha * envelope[i]
                else:
                    # Release phase
                    alpha = 1.0 / max(release_samples, 1)
                    smoothed[i] = (1 - alpha) * smoothed[i-1] + alpha * envelope[i]
        
        # Apply hold
        if hold_samples > 0:
            for i in range(len(smoothed)):
                start = max(0, i - hold_samples)
                smoothed[i] = np.max(smoothed[start:i+1])
        
        return smoothed
    
    def _stft(self, audio: np.ndarray, n_fft: int, hop_length: int, window: np.ndarray) -> np.ndarray:
        """
        Compute Short-Time Fourier Transform.
        
        Parameters:
        -----------
        audio : np.ndarray
            Input audio signal
        n_fft : int
            FFT size
        hop_length : int
            Hop size
        window : np.ndarray
            Window function
            
        Returns:
        --------
        np.ndarray
            STFT matrix
        """
        # Pad audio
        if len(audio) < n_fft:
            audio = np.pad(audio, (0, n_fft - len(audio)))
        
        # Calculate number of frames
        n_frames = 1 + (len(audio) - n_fft) // hop_length
        
        # Initialize STFT matrix
        stft = np.zeros((n_fft // 2 + 1, n_frames), dtype=complex)
        
        # Compute STFT
        for i in range(n_frames):
            start = i * hop_length
            end = start + n_fft
            
            frame = audio[start:end] * window
            stft[:, i] = np.fft.rfft(frame)
        
        return stft
    
    def _istft(self, stft: np.ndarray, hop_length: int, window: np.ndarray, original_length: int) -> np.ndarray:
        """
        Compute Inverse Short-Time Fourier Transform.
        
        Parameters:
        -----------
        stft : np.ndarray
            STFT matrix
        hop_length : int
            Hop size
        window : np.ndarray
            Window function
        original_length : int
            Original audio length
            
        Returns:
        --------
        np.ndarray
            Reconstructed audio signal
        """
        n_fft = (stft.shape[0] - 1) * 2
        n_frames = stft.shape[1]
        
        # Initialize audio buffer
        audio = np.zeros((n_frames - 1) * hop_length + n_fft)
        window_sum = np.zeros_like(audio)
        
        # Compute ISTFT
        for i in range(n_frames):
            start = i * hop_length
            end = start + n_fft
            
            frame = np.fft.irfft(stft[:, i])
            audio[start:end] += frame * window
            window_sum[start:end] += window ** 2
        
        # Normalize by window sum
        window_sum[window_sum < 1e-10] = 1.0
        audio = audio / window_sum
        
        # Trim or pad to original length
        if len(audio) > original_length:
            audio = audio[:original_length]
        elif len(audio) < original_length:
            audio = np.append(audio, np.zeros(original_length - len(audio)))
        
        return audio
    
    def update_parameters(self, **kwargs):
        """
        Update denoising parameters.
        
        Parameters:
        -----------
        **kwargs
            Parameter name-value pairs to update
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown parameter '{key}'")

