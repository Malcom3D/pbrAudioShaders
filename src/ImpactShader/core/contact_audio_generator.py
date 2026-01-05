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

import os
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from scipy import signal
from scipy.interpolate import interp1d

from ..lib.impacts_data import ImpactEvent, ContactType, ObjectContact

@dataclass
class ContactAudioGenerator:
    """Generates audio signals for contact interactions using physically-based stochastic models"""
    
    # Audio generation parameters
    sample_rate: int = 48000
    bit_depth: int = 32
    
    # Scraping/sliding parameters (from paper)
    scraping_noise_bandwidth: Tuple[float, float] = (100.0, 10000.0)  # Hz
    scraping_fractal_dimension: float = 1.5  # Fractal dimension for 1/f^β noise
    scraping_resonance_q: float = 10.0  # Q factor for resonant filter
    
    # Rolling parameters (from paper)
    rolling_resonance_range: Tuple[float, float] = (50.0, 2000.0)  # Hz
    rolling_noise_bandwidth: Tuple[float, float] = (50.0, 5000.0)  # Hz
    rolling_spectral_envelope_p: float = 500.0  # Center frequency for spectral envelope
    rolling_spectral_envelope_d: float = 100.0  # Bandwidth for spectral envelope
    
    # Impact parameters
    impact_duration_range: Tuple[float, float] = (0.001, 0.1)  # s
    impact_spectral_tilt: float = -6.0  # dB/octave
    
    def generate_contact_audio(self, event: ImpactEvent, obj_idx: int) -> np.ndarray:
        """
        Generate audio signal for a specific object in a contact event.
        
        Args:
            event: The contact event
            obj_idx: Object index
            
        Returns:
            Audio signal as numpy array at self.sample_rate
        """
        obj_contact = event.get_object_contact(obj_idx)
        if not obj_contact or not obj_contact.contacts:
            return np.zeros(int(self.sample_rate * event.get_contact_audio_duration()))
        
        # Determine dominant contact type for this object
        contact_types = obj_contact.get_contact_types()
        dominant_type = self._determine_dominant_contact_type(contact_types)
        
        # Generate audio based on contact type
        if dominant_type == ContactType.IMPACT:
            audio = self._generate_impact_audio(event, obj_contact)
        elif dominant_type == ContactType.SCRAPING:
            audio = self._generate_scraping_audio(event, obj_contact)
        elif dominant_type == ContactType.ROLLING:
            audio = self._generate_rolling_audio(event, obj_contact)
        else:  # MIXED or other
            audio = self._generate_mixed_contact_audio(event, obj_contact)
        
        # Normalize and apply envelope
        audio = self._apply_contact_envelope(audio, event, obj_contact)
        
        return audio
    
    def _determine_dominant_contact_type(self, contact_types: List[ContactType]) -> ContactType:
        """Determine dominant contact type from list"""
        if ContactType.IMPACT in contact_types:
            return ContactType.IMPACT
        elif ContactType.SCRAPING in contact_types:
            return ContactType.SCRAPING
        elif ContactType.ROLLING in contact_types:
            return ContactType.ROLLING
        else:
            return ContactType.MIXED
    
    def _generate_impact_audio(self, event: ImpactEvent, obj_contact: ObjectContact) -> np.ndarray:
        """
        Generate impact audio using physically-based model.
        
        Based on paper: impulsive excitation with spectral characteristics
        determined by contact force and material properties.
        """
        duration = event.get_contact_audio_duration()
        num_samples = int(self.sample_rate * duration)
        
        # Calculate total force magnitude
        total_force = np.linalg.norm(obj_contact.get_total_force())
        
        # Generate impact signal as filtered impulse
        impact_signal = np.zeros(num_samples)
        
        # Place impulse at appropriate time
        impulse_time = 0.01  # 10ms after start
        impulse_sample = int(impulse_time * self.sample_rate)
        if impulse_sample < num_samples:
            impact_signal[impulse_sample] = total_force * 0.1  # Scaled impulse

            # Apply resonant filter to simulate impact spectrum
            impact_signal = self._apply_impact_filter(impact_signal, total_force)
        
        return impact_signal
    
    def _apply_impact_filter(self, signal: np.ndarray, force: float) -> np.ndarray:
        """Apply impact-specific filtering based on force magnitude"""
        # Determine center frequency based on force (higher force -> higher frequency)
        f0 = 1000.0 + 2000.0 * min(force / 100.0, 1.0)  # 1-3kHz range
        
        # Create resonant filter
        Q = 5.0 + 10.0 * min(force / 100.0, 1.0)  # Higher Q for sharper impacts
        b, a = signal.iirpeak(f0, Q, fs=self.sample_rate)
        
        # Apply filter
        filtered = signal.lfilter(b, a, signal)
        
        # Apply spectral tilt (more high frequency for sharp impacts)
        tilt_filter = self._create_spectral_tilt_filter(self.impact_spectral_tilt)
        filtered = signal.lfilter(tilt_filter[0], tilt_filter[1], filtered)
        
        return filtered
    
    def _generate_scraping_audio(self, event: ImpactEvent, obj_contact: ObjectContact) -> np.ndarray:
        """
        Generate scraping/sliding audio using fractal noise and resonant filters.
        
        Based on paper: 1/f^β noise noise filtered through second-order resonant filters.
        """
        duration = event.get_contact_audio_duration()
        num_samples = int(self.sample_rate * duration)
        
        # Generate fractal noise (1/f^β noise)
        fractal_noise = self._generate_fractal_noise(
            num_samples, 
            self.scraping_fractal_dimension
        )
        
        # Calculate scraping parameters from contact data
        avg_force = np.mean([np.linalg.norm(c.force_vector) for c in obj_contact.contacts])
        avg_velocity = np.mean([np.linalg.norm(c.relative_velocity) for c in obj_contact.contacts])
        
        # Apply resonant filters based on velocity and force
        filtered_noise = self._apply_scraping_filters(
            fractal_noise, 
            avg_velocity, 
            avg_force,
            obj_contact.surface_roughness
        )
        
        # Apply amplitude modulation based on contact force variations
        if len(obj_contact.contacts) > 1:
            filtered_noise = self._apply_force_modulation(filtered_noise, obj_contact)
        
        return filtered_noise
    
    def _generate_fractal_noise(self, num_samples: int, beta: float) -> np.ndarray:
        """Generate 1/f^β noise (fractal noise)"""
        # Generate white noise
        white_noise = np.random.randn(num_samples)
        
        # Create frequency domain representation
        freq_noise = np.fft.rfft(white_noise)
        freqs = np.fft.rfftfreq(num_samples, 1/self.sample_rate)
        
        # Avoid division by zero at DC
        freqs[0] = freqs[1]
        
        # Apply 1/f^β spectral shaping
        spectral_shape = 1.0 / (freqs ** (beta / 2.0))
        freq_noise *= spectral_shape
        
        # Convert back to time domain
        fractal_noise = np.fft.irfft(freq_noise, n=num_samples)
        
        return fractal_noise
    
    def _apply_scraping_filters(self, noise: np.ndarray, velocity: float, 
                               force: float, roughness: float) -> np.ndarray:
        """Apply resonant filters for scraping sound"""
        filtered = noise.copy()
        
        # Determine number of resonant peaks based on velocity
        num_peaks = int(3 + velocity * 5)  # More peaks for higher velocity
        num_peaks = min(num_peaks, 10)  # Limit to 10 peaks
        
        # Generate resonant frequencies based on paper's model
        for i in range(num_peaks):
            # Center frequency based on velocity and roughness
            f0 = self.scraping_noise_bandwidth[0] + \
                 (self.scraping_noise_bandwidth[1] - self.scraping_noise_bandwidth[0]) * \
                 (i / num_peaks) * (1.0 + velocity * 0.5)
            
            # Q factor based on force and roughness
            Q = self.scraping_resonance_q * (1.0 + force * 0.01) * (1.0 + roughness * 100.0)
            Q = max(2.0, min(Q, 50.0))
            
            # Create and apply resonant filter
            b, a = signal.iirpeak(f0, Q, fs=self.sample_rate)
            filtered = signal.lfilter(b, a, filtered)
        
        return filtered
    
    def _generate_rolling_audio(self, event: ImpactEvent, obj_contact: ObjectContact) -> np.ndarray:
        """
        Generate rolling audio using noise-driven resonant filter with spectral envelope.
        
        Based on paper: S(ω) = 1 / sqrt((ω - p)² + d²)
        """
        duration = event.get_contact_audio_duration()
        num_samples = int(self.sample_rate * duration)
        
        # Generate noise source
        noise = np.random.randn(num_samples)
        
        # Calculate rolling parameters
        avg_velocity = np.mean([np.linalg.norm(c.relative_velocity) for c in obj_contact.contacts])
        avg_angular_vel = np.mean([
            np.linalg.norm(self._extract_angular_velocity(c)) 
            for c in obj_contact.contacts
        ])
        
        # Apply spectral envelope filter
        filtered_noise = self._apply_rolling_spectral_envelope(
            noise, 
            avg_velocity, 
            avg_angular_vel
        )
        
        # Apply resonant filter for rolling resonance
        if avg_angular_vel > 0.1:
            filtered_noise = self._apply_rolling_resonance(
                filtered_noise, 
                avg_angular_vel,
                obj_contact
            )
        
        # Apply amplitude modulation for rolling rhythm
        if avg_angular_vel > 0.5:
            filtered_noise = self._apply_rolling_modulation(
                filtered_noise, 
                avg_angular_vel,
                duration
            )
        
        return filtered_noise
    
    def _extract_angular_velocity(self, contact) -> np.ndarray:
        """Extract angular velocity from contact data"""
        # This would come from the trajectory data
        # For now, return a reasonable estimate
        return np.array([0, 0, 1.0]) * np.linalg.norm(contact.relative_velocity) * 0.1
    
    def _apply_rolling_spectral_envelope(self, noise: np.ndarray, velocity: float, 
                                        angular_vel: float) -> np.ndarray:
        """Apply spectral envelope S(ω) = 1 / sqrt((ω - p)² + d²)"""
        # Convert to frequency domain
        freq_noise = np.fft.rfft(noise)
        freqs = np.fft.rfftfreq(len(noise), 1/self.sample_rate)
        
        # Calculate spectral envelope parameters based on velocity
        p = self.rolling_spectral_envelope_p * (1.0 + velocity * 2.0)  # Center frequency
        d = self.rolling_spectral_envelope_d * (1.0 + angular_vel * 5.0)  # Bandwidth
        
        # Apply spectral envelope
        spectral_envelope = 1.0 / np.sqrt((2 * np.pi * freqs - p) ** 2 + d ** 2)
        freq_noise *= spectral_envelope
        
        # Convert back to time domain
        filtered = np.fft.irfft(freq_noise, n=len(noise))
        
        return filtered
    
    def _apply_rolling_resonance(self, signal: np.ndarray, angular_vel: float,
                                obj_contact: ObjectContact) -> np.ndarray:
        """Apply resonant filter for rolling contacts"""
        # Determine resonance frequency based on angular velocity and object size
        # For convex hull approximation
        if obj_contact.convex_hull is not None and len(obj_contact.convex_hull) > 0:
            # Estimate characteristic length from convex hull
            hull_diam = np.max(obj_contact.convex_hull) - np.min(obj_contact.convex_hull)
            char_length = np.mean(hull_diam)
        else:
            char_length = 0.1  # Default 10cm
        
        # Resonance frequency based on rolling speed and object size
        # f = v / (2πr) for a rolling object
        if char_length > 0:
            f0 = angular_vel * char_length / (2 * np.pi)
            f0 = max(self.rolling_resonance_range[0], 
                    min(f0, self.rolling_resonance_range[1]))
        else:
            f0 = 100.0  # Default 100Hz
        
        # Create resonant filter
        Q = 20.0 + angular_vel * 10.0  # Higher Q for smoother rolling
        b, a = signal.iirpeak(f0, Q, fs=self.sample_rate)
        
        filtered = signal.lfilter(b, a, signal)
        
        return filtered
    
    def _apply_rolling_modulation(self, signal: np.ndarray, angular_vel: float,
                                 duration: float) -> np.ndarray:
        """Apply amplitude modulation for rolling rhythm"""
        # Create modulation frequency based on angular velocity
        # One modulation per revolution
        mod_freq = angular_vel / (2 * np.pi)  # Hz
        
        # Create modulation signal
        t = np.linspace(0, duration, len(signal))
        modulation = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t)
        
        # Apply modulation
        modulated = signal * modulation
        
        return modulated
    
    def _generate_mixed_contact_audio(self, event: ImpactEvent, obj_contact: ObjectContact) -> np.ndarray:
        """Generate audio for mixed contact types"""
        duration = event.get_contact_audio_duration()
        num_samples = int(self.sample_rate * duration)
        
        # Generate components for different contact types
        impact_component = np.zeros(num_samples)
        scraping_component = np.zeros(num_samples)
        rolling_component = np.zeros(num_samples)
        
        # Count contact types
        type_counts = {}
        for contact in obj_contact.contacts:
            type_name = contact.contact_type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        total_contacts = len(obj_contact.contacts)
        
        # Generate weighted components
        if 'impact' in type_counts:
            impact_weight = type_counts['impact'] / total_contacts
            impact_com_component = self._generate_impact_audio(event, obj_contact) * impact_weight
        
        if 'scraping' in type_counts:
            scraping_weight = type_counts['scraping'] / total_contacts
            # # Create a modified obj_contact with only scraping contacts
            scraping_contacts = ObjectContact(obj_idx=obj_contact.obj_idx)
            for contact in obj_contact.contacts:
                if contact.contact_type == ContactType.SCRAPING:
                    scraping_contacts.add_contact(contact)
            if scraping_contacts.contacts:
                scraping_component = self._generate_scraping_audio(event, scraping_contacts) * scraping_weight
        
        if 'rolling' in type_counts:
            rolling_weight = type_counts['rolling'] / total_contacts
            # Create a modified obj_contact with only rolling contacts
            rolling_contacts = ObjectContact(obj_idx=obj_contact.obj_idx)
            for contact in obj_contact.contacts:
                if contact.contact_type == ContactType.ROLLING:
                    rolling_contacts.add_contact(contact)
            if rolling_contacts.contacts:
                rolling_component = self._generate_rolling_audio(event, rolling_contacts) * rolling_weight
        
        # Mix components
        mixed = impact_component + scraping_component + rolling_component
        
        return mixed
    
    def _apply_force_modulation(self, signal: np.ndarray, obj_contact: ObjectContact) -> np.ndarray:
        """Apply amplitude modulation based on force variations"""
        if len(obj_contact.contacts) < 2:
            return signal
        
        # Extract force magnitudes over time (simplified)
        forces = [np.linalg.norm(c.force_vector) for c in obj_contact.contacts]
        
        # Create interpolation function for force envelope
        # Assuming contacts are evenly spaced in time
        t_contacts = np.linspace(0, 1, len(forces))
        force_interp = interp1d(t_contacts, forces, kind='cubic', 
                               fill_value='extrapolate')
        
        # Create time array for signal
        t_signal = np.linspace(0, 1, len(signal))
        
        # Create modulation envelope
        modulation = force_interp(t_signal)
        modulation = (modulation - np.min(modulation)) / (np.max(modulation) - np.min(modulation) + 1e-10)
        
        # Apply modulation
        modulated = signal * (0.5 + 0.5 * modulation)
        
        return modulated
    
    def _apply_contact_envelope(self, signal: np.ndarray, event: ImpactEvent, 
                               obj_contact: ObjectContact) -> np.ndarray:
        """Apply temporal envelope to contact audio"""
        duration = len(signal) / self.sample_rate
        
        # Create envelope based on contact type and duration
        t = np.linspace(0, duration, len(signal))
        
        if event.dominant_contact_type == ContactType.IMPACT:
            # Short attack, exponential decay
            attack_time = 0.001  # 1ms attack
            decay_time = min(0.1, duration)  # 100ms decay or shorter
            
            envelope = np.zeros_like(t)
            attack_samples = int(attack_time * self.sample_rate)
            decay_samples = int(decay_time * self.sample_rate)
            
            if attack_samples > 0:
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            
            if decay_samples > attack_samples:
                decay_start = attack_samples
                decay_length = decay_samples - attack_samples
                envelope[decay_start:decay_samples] = np.exp(-5.0 * np.lininspace(0, 1, decay_length))
            
            envelope[decay_samples:] = 0
            
        elif event.dominant_contact_type == ContactType.SCRAPING:
            # Gradual attack and release for scraping
            attack_time = min(0.01, duration * 0.1)
            sustain_time = duration - attack_time * 2
            release_time = min(0.01, duration * 0.1)
            
            envelope = np.ones_like(t)
            
            if attack_time > 0:
                attack_samples = int(attack_time * self.sample_rate)
                envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
            
            if release_time > 0 and sustain_time > 0:
                release_samples = int(release_time * self.sample_rate)
                release_start = len(t) - release_samples
                envelope[release_start:] = np.linspace(1, 0, release_samples)
            
        elif event.dominant_contact_type == ContactType.ROLLING:
            # Smooth envelope for rolling
            # Slight fade in and out
            fade_time = min(0.05, duration * 0.1)
            
            envelope = np.ones_like(t)
            
            if fade_time > 0:
                fade_samples = int(fade_time * self.sample_rate)
                envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
            
        else:  # MIXED
            # Generic envelope
            envelope = np.ones_like(t)
            if duration > 0.1:
                fade_samples = int(min(0.05, duration * 0.1) * self.sample_rate)
                envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
                envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
        
        # Apply envelope
        enveloped = signal * envelope
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(enveloped))
        if max_val > 1e-10:
            enveloped = enveloped / max_val * 0.8  # -2dB headroom
        
        return enveloped
    
    def _create_spectral_tilt_filter(self, tilt_db_per_octave: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create a filter for spectral tilt"""
        # Simple first-order shelving filter approximation
        if abs(tilt_db_per_octave) < 0.1:
            return [1.0], [1.0]
        
        # Convert dB/octave to filter coefficient
        # This is a simplified approximation
        omega = 2 * np.pi * 1000.0 / self.sample_rate  # 1kHz reference
        gain = 10 ** (tilt_db_per_octave / 20.0 / 3.0)  # Approximate for 3 octaves
        
        # Create simple shelving filter
        if tilt_db_per_octave > 0:
            # Boost high frequencies
            b = [gain, 0]
            a = [1, gain - 1]
        else:
            # Cut high frequencies
            b = [1, gain - 1]
            a = [gain, 0]
        
        return b, a
    
    def save_contact_audio(self, audio: np.ndarray, event: ImpactEvent, 
                          obj_idx: int, output_dir: str) -> str:
        """Save contact audio to WAV file"""
        import soundfile as sf
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename
        filename = f"contact_event{event.idx:04d}_obj{obj_idx}_" \
                  f"{event.dominant_contact_type.value}.wav"
        filepath = os.path.join(output_dir, filename)
        
        # Save as WAV file
        sf.write(filepath, audio, self.sample_rate, subtype='FLOAT')
        
        print(f"Saved contact audio: {filepath}")
        return filepath
