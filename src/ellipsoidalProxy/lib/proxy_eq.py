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
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from pbrAudioCommon import EntityManager
from pbrAudioCommon import debug_print, set_debug, set_debug_prefix

@dataclass
class ProxyEqualizer:
    """
    Dynamic frequency equalizer for proxy synth output.
    
    Adjusts frequency response based on contact type and force characteristics.
    Uses SIMD-optimized FFT processing.
    """
    entity_manager: EntityManager
    
    sample_rate: int = 48000
    fft_size: int = 4096
    hop_size: int = 1024
    
    # Equalization curves for each contact type
    # These are applied as frequency-dependent gains
    eq_curves: Dict[int, np.ndarray] = field(default_factory=dict)
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.sample_rate = config.system.sample_rate

        set_debug(config.system.debug)
        set_debug_prefix(self.__class__.__name__)

        # Initialize default EQ curves
        self._init_default_curves()
        
        # Pre-compute FFT window
        self._window = np.hanning(self.fft_size)
        
        # Pre-compute frequency axis
        self._freqs = np.fft.rfftfreq(self.fft_size, 1/self.sample_rate)
    
    def _init_default_curves(self):
        """Initialize default EQ curves for each contact type."""
        n_freqs = self.fft_size // 2 + 1
        freqs = np.fft.rfftfreq(self.fft_size, 1/self.sample_rate)
        
        # Impact: emphasize from mid-low frequencies
        impact_curve = np.ones(n_freqs)
        impact_curve[freqs < 250] *= np.exp(-(250 - freqs[freqs < 250]) / 5000)
        self.eq_curves[0] = impact_curve
        
        # Sliding: emphasize mid frequencies
        sliding_curve = np.ones(n_freqs)
        sliding_curve[freqs < 3000] *= 0.1
        sliding_curve[freqs > 9000] *= np.exp(-(freqs[freqs > 9000] - 9000) / 8000)
        self.eq_curves[1] = sliding_curve
        
        # Scraping: emphasize high frequencies
        scraping_curve = np.ones(n_freqs)
        scraping_curve[freqs < 2000] *= np.exp(-(2000 - freqs[freqs < 2000]) / 3000)
        scraping_curve[freqs > 10000] *= 0.1
        self.eq_curves[2] = scraping_curve
        
        # Rolling: emphasize high frequencies
        rolling_curve = np.ones(n_freqs)
        rolling_curve[freqs < 6500] *= np.exp(-(6500 - freqs[freqs < 6500]) / 3000)
        self.eq_curves[3] = rolling_curve

        # Rolling sound: emphasize low frequencies
        rolling_sound_curve = np.ones(n_freqs)
        rolling_sound_curve[freqs > 500] *= np.exp(-(freqs[freqs > 500] - 500) / 8000)
        self.eq_curves[4] = rolling_sound_curve

        # Sliding sound: emphasize low frequencies
        sliding_sound_curve = np.ones(n_freqs)
        sliding_sound_curve[freqs < 250] *= np.exp(-(250 - freqs[freqs < 250]) / 3000)
        sliding_sound_curve[freqs > 3000] *= np.exp(-(freqs[freqs > 3000] - 3000) / 8000)
        self.eq_curves[5] = sliding_sound_curve

        # Scraping sound: emphasize low frequencies
        scraping_sound_curve = np.ones(n_freqs)
        sliding_sound_curve[freqs < 150] *= np.exp(-(150 - freqs[freqs < 150]) / 3000)
        scraping_sound_curve[freqs > 5000] *= np.exp(-(freqs[freqs > 5000] - 5000) / 8000)
        self.eq_curves[6] = scraping_sound_curve

    def apply_equalization(self, audio: np.ndarray, contact_type: int, force: np.ndarray = None) -> np.ndarray:
        """
        Apply dynamic frequency equalization to audio.
        
        Parameters:
        -----------
        audio : np.ndarray
            Input audio signal
        contact_type : int
            Contact type (0=impact, 1=sliding, 2=scraping, 3=rolling, 4=rolling_sound)
        force : np.ndarray, optional
            Force signal for dynamic EQ adjustment
        
        Returns:
        --------
        np.ndarray : Equalized audio
        """
        audio_values = np.count_nonzero(audio)
        
        if audio_values == 0:
            return audio

        n_samples = audio.shape[0]
        
        # Get base EQ curve for this contact type
        base_curve = self.eq_curves.get(contact_type, np.ones(self.fft_size // 2 + 1))
        
        # If force is provided, compute dynamic EQ adjustments
        if force is not None and len(force) > 0:
            # Compute force envelope
            force_env = self._compute_force_envelope(force, n_samples)
            
            # Process in blocks with overlap-add
            output = self._process_with_dynamic_eq(audio, base_curve, force_env)
        else:
            # Static EQ - process entire signal
            output = self._apply_static_eq(audio, base_curve)
        
        return output
    
    def _compute_force_envelope(self, force: np.ndarray, n_samples: int) -> np.ndarray:
        """Compute smooth force envelope for dynamic EQ."""
        # Resample force to match audio length
        if len(force) != n_samples:
            # Use linear interpolation
            x_old = np.linspace(0, 1, len(force))
            x_new = np.linspace(0, 1, n_samples)
            force_env = np.interp(x_new, x_old, force)
        else:
            force_env = force.copy()
        
        # Smooth the envelope
        window_size = max(3, int(self.sample_rate * 0.01))  # 10ms window
        kernel = np.ones(window_size) / window_size
        force_env = np.convolve(force_env, kernel, mode='same')
        
        # Normalize to 0-1 range
        max_force = np.max(np.abs(force_env))
        if max_force > 0:
            force_env = force_env / max_force
        
        return force_env
    
    def _process_with_dynamic_eq(self, audio: np.ndarray, base_curve: np.ndarray, force_env: np.ndarray) -> np.ndarray:
        """
        Process audio with dynamic EQ based on force envelope.
        
        Uses overlap-add with time-varying frequency response.
        """
        n_samples = audio.shape[0]
        output = np.zeros(n_samples + self.fft_size)
        
        # Process in blocks
        n_blocks = int(np.ceil(n_samples / self.hop_size))
        
        for block_idx in range(n_blocks):
            start = block_idx * self.hop_size
            end = min(start + self.fft_size, n_samples)
            block_len = end - start
            
            if block_len <= 0:
                continue
            
            # Extract block
            block = audio[start:end]
            
            # Apply window
            windowed = block * self._window[:block_len]
            
            # Pad to FFT size
            padded = np.zeros(self.fft_size)
            padded[:block_len] = windowed
            
            # FFT
            block_fft = np.fft.rfft(padded)
            
            # Compute dynamic EQ curve for this block
            # Average force in this block
            block_force = np.mean(force_env[start:end]) if end > start else 0
            
            # Adjust EQ based on force
            # Higher force = more high-frequency content
            dynamic_curve = base_curve.copy()
            if block_force > 0.1:
                # Boost high frequencies with force
                high_freq_boost = 1.0 + 0.5 * block_force
                dynamic_curve[self._freqs > 2000] *= high_freq_boost
            
            # Apply EQ
            block_fft *= dynamic_curve
            
            # Inverse FFT
            result = np.fft.irfft(block_fft, n=self.fft_size)
            
            # Apply window again (for overlap-add)
            result *= self._window
            
            # Overlap-add
            output[start:start + self.fft_size] += result
        
        # Trim to original length
        output = output[:n_samples]
        
        return output
    
    def _apply_static_eq(self, audio: np.ndarray, eq_curve: np.ndarray) -> np.ndarray:
        """
        Apply static EQ to entire signal using overlap-add for long signals.
    
        Parameters:
        -----------
        audio : np.ndarray
            Input audio signal
        eq_curve : np.ndarray
            Frequency-domain EQ curve (n_fft/2 + 1)
    
        Returns:
        --------
        np.ndarray : Equalized audio
        """
        n_samples = audio.shape[0]
    
        # If signal is shorter than FFT size, process directly
        if n_samples <= self.fft_size:
            return self._apply_static_eq_block(audio, eq_curve)
    
        # For longer signals, use overlap-add
        output = np.zeros(n_samples + self.fft_size, dtype=np.float32)
    
        # Process in blocks with 50% overlap
        hop_size = self.fft_size // 2
    
        for start in range(0, n_samples, hop_size):
            end = min(start + self.fft_size, n_samples)
            block_len = end - start
        
            # Extract block and pad if necessary
            block = np.zeros(self.fft_size, dtype=np.float32)
            block[:block_len] = audio[start:end]
        
            # Apply window to reduce artifacts
            windowed = block * self._window
        
            # FFT
            block_fft = np.fft.rfft(windowed)
        
            # Apply EQ curve
            block_fft *= eq_curve
        
            # Inverse FFT
            result = np.fft.irfft(block_fft, n=self.fft_size)
        
            # Apply window again (for overlap-add)
            result *= self._window
        
            # Overlap-add to output
            output[start:start + self.fft_size] += result
    
        # Trim to original length
        output = output[:n_samples]
    
        return output

    def _apply_static_eq_block(self, audio: np.ndarray, eq_curve: np.ndarray) -> np.ndarray:
        """
        Apply static EQ to a single block (shorter than FFT size).
    
        Parameters:
        -----------
        audio : np.ndarray
            Input audio block
        eq_curve : np.ndarray
            Frequency-domain EQ curve (n_fft/2 + 1)
    
        Returns:
        --------
        np.ndarray : Equalized audio
        """
        n_samples = audio.shape[0]
    
        # Pad to FFT size
        padded = np.zeros(self.fft_size, dtype=np.float32)
        padded[:min(n_samples, self.fft_size)] = audio[:min(n_samples, self.fft_size)]
    
        # Apply window
        windowed = padded * self._window
    
        # FFT
        audio_fft = np.fft.rfft(windowed)
    
        # Apply EQ
        audio_fft *= eq_curve
    
        # Inverse FFT
        result = np.fft.irfft(audio_fft, n=self.fft_size)
    
        # Apply window again
        result *= self._window
    
        # Trim to original length
        output = result[:n_samples]
    
        return output

    def update_eq_curve(self, contact_type: int, curve: np.ndarray) -> None:
        """Update EQ curve for a contact type."""
        self.eq_curves[contact_type] = curve
    
    def get_eq_curve(self, contact_type: int) -> np.ndarray:
        """Get EQ curve for a contact type."""
        return self.eq_curves.get(contact_type, np.ones(self.fft_size // 2 + 1))

