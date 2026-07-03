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

"""

Post-processing module for rigid body synthesized audio tracks.
Uses audio-forces as dynamic reference signals to denoise, smooth,
phase align, amplify, and blend rendered sound tracks.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import soundfile as sf
import json

from physicsSolver import EntityManager

@dataclass
class PostProcess:
    """
    Post-processor for rigid body synthesized audio.
    
    Uses audio-force signals as dynamic references to:
    - Denoise rendered tracks while preserving transient detail
    - Smooth artifacts during quiet sections
    - Phase align and blend multiple tracks
    - Apply dynamic amplification based on force envelopes
    """
    
    entity_manager: EntityManager
    
    def __post_init__(self):
        config = self.entity_manager.get('config')
        self.system_config = config.system
        self.sample_rate = self.system_config.sample_rate
        self.cache_path = self.system_config.cache_path
        self.output_dir = f"{self.cache_path}/modal_player"
        self.audio_force_dir = f"{self.cache_path}/audio_force"
        
        # Pre-compute filter coefficients
        self._init_filters()
    
    def _init_filters(self):
        """Initialize filters used in post-processing."""
        # Noise gate envelope follower
        self.gate_attack = int(self.postprocess.noise_gate_threshold_db * self.sample_rate / 1000)
        self.gate_release = int(50 * self.sample_rate / 1000)  # 50ms release
        
        # Smoothing filter (low-pass)
        smoothing_cutoff = 1000.0 / (self.postprocess.smoothing_window_ms * 2.0)
        self.smoothing_b, self.smoothing_a = signal.butter(
            2, smoothing_cutoff / (self.sample_rate / 2), btype='low'
        )
        
        # De-esser / high-frequency reduction
        self.deesser_b, self.deesser_a = signal.butter(2, 8000.0 / (self.sample_rate / 2), btype='low')
    
    def process_object(self, obj_name: str, obj_idx: int) -> Dict[str, np.ndarray]:
        """
        Process all tracks for a single object.
        
        Args:
            obj_name: Object name
            obj_idx: Object index
            
        Returns:
            Dictionary of processed tracks
        """
        if self.postprocess.verbose:
            print(f"PostProcess: Processing {obj_name} (idx={obj_idx})")
        
        # Load rendered tracks
        rendered_tracks = self._load_rendered_tracks(obj_name)
        
        # Load force reference signals
        force_signals = self._load_force_signals(obj_name)
        
        if not rendered_tracks:
            print(f"PostProcess: No rendered tracks found for {obj_name}")
            return {}
        
        # Process each track
        processed_tracks = {}
        
        for track_name, rendered_signal in rendered_tracks.items():
            # Get corresponding force signal
            force_signal = self._get_force_for_track(track_name, force_signals)
            
            # Apply post-processing
            processed = self._process_track(rendered_signal, force_signal, track_name)
            processed_tracks[track_name] = processed
        
        # Blend tracks together if enabled
        if self.postprocess.blend_enabled and len(processed_tracks) > 1:
            processed_tracks = self._blend_tracks(processed_tracks)
        
        # Save processed tracks
        self._save_processed_tracks(obj_name, processed_tracks)
        
        return processed_tracks
    
    def _load_rendered_tracks(self, obj_name: str) -> Dict[str, np.ndarray]:
        """Load rendered audio tracks from modal_player output."""
        tracks = {}
        
        track_types = ['rigidbody', 'resonance', 'sliding', 'scraping', 'rolling']
        
        for track_type in track_types:
            filepath = f"{self.output_dir}/{obj_name}_{track_type}.raw"
            if os.path.exists(filepath):
                try:
                    data = np.fromfile(filepath, dtype=np.float32)
                    tracks[track_type] = data
                    if self.postprocess.verbose:
                        print(f"  Loaded {track_type}: {len(data)} samples")
                except Exception as e:
                    print(f"  Warning: Could not load {filepath}: {e}")
        
        return tracks
    
    def _load_force_signals(self, obj_name: str) -> Dict[str, np.ndarray]:
        """Load audio-force reference signals."""
        forces = {}
        
        force_types = ['impact', 'sliding', 'scraping', 'rolling', 'coupling_strength']
        
        for force_type in force_types:
            filepath = f"{self.audio_force_dir}/{obj_name}_{force_type}.raw"
            if os.path.exists(filepath):
                try:
                    data = np.fromfile(filepath, dtype=np.float32)
                    forces[force_type] = data
                except Exception as e:
                    if self.postprocess.verbose:
                        print(f"  Force {force_type} not available: {e}")
        
        return forces
    
    def _get_force_for_track(self, track_name: str, force_signals: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
        """Get the appropriate force signal for a track type."""
        mapping = {
            'rigidbody': 'impact',     # Rigidbody uses impact force primarily
            'resonance': 'impact',     # Resonance also uses impact
            'sliding': 'sliding',
            'scrapraping': 'scraping',
            'rolling': 'rolling'
        }
        
        force_type = mapping.get(track_name)
        if force_type and force_type in force_signals:
            return force_signals[force_type]
        
        # Fallback: use coupling strength as general force envelope
        if 'coupling_strength' in force_signals:
            return force_signals['coupling_strength']
        
        return None
    
    def _process_track(self, 
                       signal_data: np.ndarray, 
                       force_signal: Optional[np.ndarray],
                       track_name: str) -> np.ndarray:
        """
        Process a single track with all enabled post-processing steps.
        
        Args:
            signal_data: Rendered audio signal
            force_signal: Force reference signal (or None)
            track_name: Name of the track for logging
            
        Returns:
            Processed signal
        """
        if len(signal_data) == 0:
            return signal_data
        
        # Make a copy to avoid modifying original
        processed = signal_data.copy().astype(np.float64)
        
        # Ensure force signal matches length
        if force_signal is not None:
            force_signal = self._align_signals(processed, force_signal)
        else:
            force_signal = np.ones_like(processed) * 0.5  # Neutral force
        
        # Step 1: Apply noise gate using force envelope
        if self.postprocess.dynamic_denoise_enabled:
            processed = self._apply_noise_gate(processed, force_signal)
        
        # Step 2: Spectral noise reduction with force force reference
        if self.postprocess.dynamic_denoise_enabled:
            processed = self._spectral_denoise(processed, force_signal)
        
        # Step 3: Adaptive smoothing
        if self.postprocess.smoothing_enabled:
            processed = self._adaptive_smoothing(processed, force_signal)
        
        # Step 4: Phase alignment with force envelope
        if self.postprocess.phase_align_enabled:
            processed = self._phase_align(processed, force_signal)
        
        # Step 5: Dynamic amplification
        processed = self._dynamic_amplify(processed, force_signal, track_name)
        
        # Step 6: Apply force-weighted blend (dry/wet)
        if self.postprocess.blend_enabled:
            processed = self._force_weighted_blend(signal_data, processed, force_signal)
        
        # Final cleanup
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalize if enabled
        if self.postprocess.normalize_output:
            max_val = np.max(np.abs(processed))
            if max_val > 0:
                processed = processed / max_val * 0.95  # Leave headroom
        
        return processed.astype(np.float32)
    
    def _align_signals(self, signal_a: np.ndarray, signal_b: np.ndarray) -> np.ndarray:
        """Align force signal to match audio signal length."""
        if len(signal_b) == len(signal_a):
            return signal_b
        elif len(signal_b) > len(signal_a):
            return signal_b[:len(signal_a)]
        else:
            # Pad with zeros or repeat
            padded = np.zeros(len(signal_a))
            padded[:len(signal_b)] = signal_b
            return padded
    
    def _apply_noise_gate(self, signal: np.ndarray, force: np.ndarray) -> np.ndarray:
        """
        Apply noise gate using force envelope as sidechain.
        
        When force is below threshold, gate closes more aggressively.
        When force is high, gate opens to preserve transients.
        """
        # Compute force envelope
        force_env = np.abs(force)
        force_env = gaussian_filter1d(force_env, sigma=2)
        
        # Normalize force envelope
        if np.max(force_env) > 0:
            force_env = force_env / np.max(force_env)
        
        # Compute adaptive threshold based on force
        threshold_linear = 10 ** (self.postprocess.noise_gate_threshold_db / 20.0)
        
        # Gate curve: when force is high, threshold lowers (gate opens)
        adaptive_threshold = threshold_linear * (1.0 - 0.9 * force_env)
        
        # Apply gate
        signal_abs = np.abs(signal)
        gate = np.ones_like(signal)
        
        # Smooth gate transitions
        smooth_window = int(0.001 * self.sample_rate)  # 1ms
        if smooth_window > 0:
            # Create smooth gate envelope
            gate_smooth = np.zeros_like(signal)
            
            for i in range(len(signal)):
                if signal_abs[i] > adaptive_threshold[i]:
                    gate_smooth[i] = 1.0
                else:
                    gate_smooth[i] = 0.0
            
            # Smooth the gate
            gate_smooth = gaussian_filter1d(gate_smooth, sigma=smooth_window)
            gate = gate_smooth
        
        return signal * gate
    
    def _spectral_denoise(self, signal: np.ndarray, force: np.ndarray) -> np.ndarray:
        """
        Spectral noise reduction using force signal as reference.
        
        Uses STFT to reduce noise in frequency bins where force is low.
        """
        # STFT parameters
        n_fft = 2048
        hop_length = n_fft // 4
        
        # Compute STFT
        f, t, Zxx = signal.stft(signal, fs=self.sample_rate, 
                                 nperseg=n_fft, noverlap=n_fft - hop_length)
        
        # Compute force envelope at STFT time points
        force_resampled = np.interp(
            np.linspace(0, len(signal), Zxx.shape[1]),
            np.arange(len(force)),
            force
        )
        force_env = np.abs(force_resampled)
        if np.max(force_env) > 0:
            force_env = force_env / np.max(force_env)
        
        # Compute noise floor estimate from quiet sections
        quiet_mask = force_env < 0.1
        if np.any(quiet_mask):
            noise_floor = np.mean(np.abs(Zxx[:, quiet_mask])**2, axis=1)
            noise_floor = np.sqrt(noise_floor)
        else:
            noise_floor = np.median(np.abs(Zxx), axis=1) * 0.1
        
        # Apply spectral subtraction with force-dependent reduction
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        
        # Reduction strength varies with force
        reduction_strength = self.postprocess.spectral_reduction_strength * (1.0 - 0.8 * force_env)
        
        # Spectral subtraction
        magnitude_clean = np.maximum(
            magnitude - reduction_strength * noise_floor[:, np.newaxis],
            0.0
        )
        
        # Reconstruct signal
        Zxx_clean = magnitude_clean * np.exp(1j * phase)
        _, processed = signal.istft(Zxx_clean, fs=self.sample_rate, 
                                     nperseg=n_fft, noverlap=n_fft - hop_length)
        
        # Match length
        if len(processed) > len(signal):
            processed = processed[:len(signal)]
        elif len(processed) < len(signal):
            processed = np.pad(processed, (0, len(signal) - len(processed)))
        
        return processed
    
    def _adaptive_smoothing(self, signal: np.ndarray, force: np.ndarray) -> np.ndarray:
        """
        Adaptive smoothing that preserves transients during high-force events.
        
        Smooths more during quiet sections, less during active sections.
        """
        # Compute force envelope
        force_env = np.abs(force)
        if np.max(force_env) > 0:
            force_env = force_env / np.max(force_env)
        
        # Smooth force envelope for gradual transitions
        force_env = gaussian_filter1d(force_env, sigma=10)
        
        # Compute adaptive smoothing factor (0 = no smoothing, 1 = full smoothing)
        smoothing_factor = 1.0 - force_env
        smoothing_factor = np.clip(smoothing_factor, 0.0, 1.0)
        
        # Apply time-varying low-pass filter
        processed = np.zeros_like(signal)
        
        # Use exponential moving average with adaptive alpha
        alpha_min = 0.1  # Very little smoothing (fast response)
        alpha_max = 0.9  # Heavy smoothing
        
        alpha = alpha_min + (alpha_max - alpha_min) * smoothing_factor
        
        # Apply EMA
        processed[0] = signal[0]
        for i in range(1, len(signal)):
            processed[i] = (1 - alpha[i]) * signal[i] + alpha[i] * processed[i-1]
        
        return processed
    
    def _phase_align(self, signal: np.ndarray, force: np.ndarray) -> np.ndarray:
        """
        Phase align signal using force envelope as reference.
        
        Ensures transient peaks align with force onset for more natural sound.
        """
        # Detect transients in force signal
        force_diff = np.diff(np.abs(force), prepend=0)
        force_onset = force_diff > np.std(force_diff) * 2.0
        
        # Detect transients in audio signal
        signal_diff = np.diff(np.abs(signal), prepend=0)
        signal_onset = signal_diff > np.std(signal_diff) * 2.0
        
        # Find onset indices
        force_onset_idx = np.where(force_onset)[0]
        signal_onset_idx = np.where(signal_onset)[0]
        
        if len(force_onset_idx) == 0 or len(signal_onset_idx) == 0:
            return signal  # No transients to align
        
        processed = signal.copy()
        
        # Align each force onset to nearest audio onset
        for f_idx in force_onset_idx:
            # Find nearest audio onset within a window
            window = int(0.01 * self.sample_rate)  # 10ms window
            nearby = signal_onset_idx[
                (signal_onset_idx > f_idx - window) & 
                (signal_onset_idx < f_idx + window)
            ]
            
            if len(nearby) > 0:
                # Find closest audio onset
                nearest = nearby[np.argmin(np.abs(nearby - f_idx))]
                shift = nearest - f_idx
                
                if abs(shift) < window // 2:
                    # Apply phase shift using crossfade
                    crossfade_len = min(self.postprocess.crossfade_samples, 
                                        len(signal) - max(f_idx, nearest))
                    
                    if crossfade_len > 0:
                        # Create crossfade window
                        fade_in = np.linspace(0, 1, crossfade_len)
                        fade_out = 1 - fade_in
                        
                        # Apply shift with crossfade
                        shifted = np.roll(signal, -shift)
                        
                        if shift > 0:
                            # Signal needs to start earlier
                            processed[f_idx:f_idx + crossfade_len] = (
                                signal[f_idx:f_idx + crossfade_len] * fade_out +
                                shifted[f_idx:f_idx + crossfade_len] * fade_in
                            )
                        else:
                            # Signal needs to start later
                            end = f_idx + crossfade_len
                            processed[f_idx:end] = (
                                signal[f_idx:end] * fade_out +
                                shifted[f_idx:end] * fade_in
                            )
        
        return processed
    
    def _dynamic_amplify(self, 
                         signal: np.ndarray, 
                         force: np.ndarray, 
                         track_name: str) -> np.ndarray:
        """
        Dynamic amplification based on force envelope.
        
        Applies gain that varies with force level, with compression
        to prevent excessive dynamics.
        """
        # Compute force envelope
        force_env = np.abs(force)
        if np.max(force_env) > 0:
            force_env = force_env / np.max(force_env)
        
        # Smooth force envelope
        force_env = gaussian_filter1d(force_env, sigma=5)
        
        # Target RMS for this track type
        track_rms_targets = {
            'rigidbody': 0.15,
            'resonance': 0.08,
            'sliding': 0.10,
            'scraping': 0.12,
            'rolling': 0.08
        }
        target_rms = track_rms_targets.get(track_name, self.postprocess.target_rms)
        
        # Compute current RMS
        current_rms = np.sqrt(np.mean(signal**2))
        
        # Base gain to reach target RMS
        if current_rms > 0:
            base_gain = target_rms / current_rms
        else:
            base_gain = 1.0
        
        # Limit gain
        max_gain_linear = 10 ** (self.postprocess.max_gain_db / 20.0)
        base_gain = np.clip(base_gain, 0.1, max_gain_linear)
        
        # Dynamic gain envelope based on force
        # High force = more gain, low force = less gain
        force_gain = 0.3 + 0.7 * force_env
        
        # Apply compression
        compression = self.postprocess.dynamic_range_compression
        
        # Compress the gain envelope
        gain_db = 20 * np.log10(force_gain + 1e-10)
        gain_db_compcompressed = gain_db * (1.0 - compression) + np.mean(gain_db) * compression
        force_gain_compressed = 10 ** (gain_db_compressed / 20.0)
        
        # Combine gains
        total_gain = base_gain * force_gain_compressed
        
        # Smooth gain to prevent artifacts
        total_gain = gaussian_filter1d(total_gain, sigma=3)
        
        # Apply gain
        processed = signal * total_gain
        
        return processed
    
    def _force_weighted_blend(self, 
                               dry: np.ndarray, 
                               wet: np.ndarray, 
                               force: np.ndarray) -> np.ndarray:
        """
        Blend dry and wet signals using force envelope.
        
        During high-force events, use more processed signal.
        During quiet sections, use more dry signal to preserve natural decay.
        """
        # Compute force envelope
        force_env = np.abs(force)
        if np.max(force_env) > 0:
            force_env = force_env / np.max(force_env)
        
        # Smooth force envelope
        force_env = gaussian_filter1d(force_env, sigma=10)
        
        # Blend factor varies with force
        # High force: more wet, Low force: more dry
        blend_factor = self.postprocess.dry_wet_mix * (0.5 + 0.5 * force_env)
        blend_factor = np.clip(blend_factor, 0.0, 1.0)
        
        # Apply blend
        processed = (1 - blend_factor) * dry + blend_factor * wet
        
        return processed
    
    def _blend_tracks(self, tracks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Blend multiple tracks together to create cohesive output.
        
        Ensures tracks don't cancel out and frequencies are balanced.
        """
        if 'rigidbody' not in tracks:
            return tracks
        
        # Get the main track (rigidbody)
        main = tracks['rigidbody'].copy()
        
        # Add other tracks with appropriate levels
        track_levels = {
            'resonance': 0.3,
            'sliding': 0.5,
            'scraping': 0.4,
            'rolling': 0.3
        }
        
        for track_name, level in track_levels.items():
            if track_name in tracks and len(tracks[track_name]) > 0:
                # Align length
                track = tracks[track_name]
                if len(track) < len(main):
                    track = np.pad(track, (0, len(main) - len(track)))
                elif len(track) > len(main):
                    track = track[:len(main)]
                
                # Add with crossfade to prevent clicks
                crossfade = min(self.postprocess.crossfade_samples, len(main))
                fade_in = np.linspace(0, 1, crossfade)
                fade_out = np.linspace(1, 0, crossfade)
                
                main[:crossfade] += track[:crossfade] * level * fade_in
                main[-crossfade:] += track[-crossfade:] * level * fade_out
                main[crossfade:-crossfade] += track[crossfade:-crossfade] * level
        
        # Update tracks with blended version
        tracks['blended'] = main
        
        return tracks
    
    def _save_processed_tracks(self, obj_name: str, tracks: Dict[str, np.ndarray]):
        """Save processed tracks to disk."""
        output_dir = f"{self.cache_path}/post_processed"
        os.makedirs(output_dir, exist_ok=True)
        
        project_data = {
            'object_name': obj_name,
            'sample_rate': self.sample_rate,
            'tracks': []
        }
        
        for track_name_name, signal_data in tracks.items():
            if len(signal_data) == 0:
                continue
            
            # Save as RAW file
            filename = f"{obj_name}_{track_name}_post.raw"
            filepath = f"{output_dir}/{filename}"
            
            signal_data.astype(np.float32).tofile(filepath)
            
            project_data['tracks'].append({
                'name': track_name,
                'file': filename,
                'channels': 1,
                'duration': len(signal_data) / self.sample_rate,
                'sample_rate': self.sample_rate
            })
        
        # Save project file
        project_file = f"{output_dir}/{obj_name}_post.json"
        with open(project_file, 'w') as f f:
            json.dump(project_data, f, indent=2)
        
        if self.postprocess.verbose:
            print(f"PostProcess: Saved processed tracks to {output_dir}")
    
    def process_all_objects(self) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Process all objects in the scene.
        
        Returns:
            Dictionary mapping object names to their processed tracks
        """
        config = self.entity_manager.get('config')
        results = {}
        
        for obj in config.objects:
            if not obj.static:  # Only process dynamic objects
                tracks = self.process_object(obj.name, obj.idx)
                if tracks:
                    results[obj.name] = tracks
        
        return results
