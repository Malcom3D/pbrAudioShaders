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
from dataclasses import dataclass
from collections import deque
import warnings

@dataclass
class ContactClassifier:
    window_size: int = 10
    velocity_adaptation_rate: float = 0.1
    force_adaptation_rate: float = 0.05

    def __post_init__(self):
        """
        Contact classifier with adaptive thresholds.
        
        Args:
            window_size: Number of recent frames to consider for threshold adaptation
            velocity_adaptation_rate: How quickly velocity thresholds adapt (0-1)
            force_adaptation_rate: How quickly force thresholds adapt (0-1)
        """
        # Initialize adaptive adaptive thresholds
        self.velocity_threshold = 0.1  # Initial threshold in m/s
        self.force_ratio_threshold = 0.3  # Initial tangential/normal force ratio
        self.velocity_variance_threshold = 0.01  # Initial variance threshold
        
        # History buffers for adaptation
        self.velocity_history = deque(maxlen=self.window_size)
        self.force_ratio_history = deque(maxlen=self.window_size)
        self.velocity_variance_history = deque(maxlen=self.window_size)
        
        # Statistical accumulators
        self.mean_velocity = 0.0
        self.mean_force_ratio = 0.0
        self.mean_velocity_variance = 0.0
        
    def _calculate_contact_features(self, sample_idx:int, trajectory: Any, force: Any) -> Dict[str, float]:
        """
        Calculate features for contact classification.
        
        Returns:
            Dictionary of calculated features
        """
        features = {}
        
        # 1. Tangential velocity magnitude (sliding speed)
        normal_component = np.dot(force.get_relative_velocity(sample_idx), trajectory.get_normals(sample_idx)) * trajectory.get_normals(sample_idx)
        tangential_velocity = force.get_tangential_velocity(sample_idx)
        features['tangential_velocity_mag'] = np.linalg.norm(tangential_velocity)
        
        # 2. Force ratio (tangential/normal)
        if force.get_normal_force_magnitude(sample_idx) > 1e-10:  # Avoid division by zero
            features['force_ratio'] = force.get_tangential_force_magnitude(sample_idx) / force.get_normal_force_magnitude(sample_idx)
        else:
            features['force_ratio'] = 0.0
            
        # 3. Velocity direction consistency (variance over recent frames)
        features['velocity_direction'] = tangential_velocity / (features['tangential_velocity_mag'] + 1e-10)
        
        # 4. Contact angle (between velocity and normal)
        if np.linalg.norm(force.get_relative_velocity(sample_idx)) > 1e-10:
            cos_angle = np.abs(np.dot(force.get_relative_velocity(sample_idx), trajectory.get_normals(sample_idx))) / \
                       (np.linalg.norm(force.get_relative_velocity(sample_idx)) * np.linalg.norm(trajectory.get_normals(sample_idx)))
            features['contact_angle'] = np.arccos(np.clip(cos_angle, -1.0, 1.0))
        else:
            features['contact_angle'] = 0.0
            
        # 5. Energy ratio (tangential kinetic energy / normal kinetic energy)
        normal_kinetic = 0.5 * force.get_normal_velocity(sample_idx)**2
        tangential_kinetic = 0.5 * features['tangential_velocity_mag']**2
        if normal_kinetic > 1e-10:
            features['energy_ratio'] = tangential_kinetic / normal_kinetic
        else:
            features['energy_ratio'] = 0.0 if tangential_kinetic < 1e-10 else 1e10
            
        return features
    
    def _update_thresholds(self, features: Dict[str, float]):
        """
        Dynamically update classification thresholds based on recent history.
        """
        # Update history buffers
        self.velocity_history.append(features['tangential_velocity_mag'])
        self.force_ratio_history.append(features['force_ratio'])
        
        # Calculate statistics if we have enough data
        if len(self.velocity_history) >= 3:
            # Update velocity threshold (adaptive to scene dynamics)
            current_mean_vel = np.mean(self.velocity_history)
            self.mean_velocity = (1 - self.velocity_adaptation_rate) * self.mean_velocity + self.velocity_adaptation_rate * current_mean_vel
            vel_std = np.std(self.velocity_history)
            self.velocity_threshold = max(0.01, self.mean_velocity + 0.5 * vel_std)
            
            # Update force ratio threshold
            current_mean_ratio = np.mean(self.force_ratio_history)
            self.mean_force_ratio = (1 - self.force_adaptation_rate) * self.mean_force_ratio + self.force_adaptation_rate * current_mean_ratio
            ratio_std = np.std(self.force_ratio_history)
            self.force_ratio_threshold = max(0.1, min(0.8, self.mean_force_ratio + 0.3 * ratio_std))
            
            # Update velocity variance threshold
            if len(self.velocity_history) >= 5:
                recent_velocities = list(self.velocity_history)[-5:]
                variance = np.var(recent_velocities)
                self.velocity_variance_history.append(variance)
                if len(self.velocity_variance_history) >= 3:
                    self.mean_velocity_variance = np.mean(self.velocity_variance_history)
                    self.velocity_variance_threshold = max(0.001, self.mean_velocity_variance * 2.0)
    
    def compute(self, sample_idx: int, trajectory: Any, force: Any, previous_classification: Optional[str] = None) -> Tuple[str, Dict[str, float]]:
        """
        Classify a contact event as scraping or sliding.
        
        Args:
            contact: Contact point data
            previous_classification: Previous classification for hysteresis
            time_step: Simulation time step in seconds
            
        Returns:
            Tuple of (classification, confidence_scores)
        """
        # Calculate features
        features = self._calculate_contact_features(sample_idx, trajectory, force)
        
        # Update adaptive thresholds
        self._update_thresholds(features)
        
        # Calculate individual scores (0-1 where higher means more likely scraping)
        scores = {}
        
        # 1. Velocity magnitude score (low velocity suggests scraping)
        vel_score = 1.0 - min(1.0, features['tangential_velocity_mag'] / max(self.velocity_threshold, 0.01))
        scores['velocity_score'] = vel_score
        
        # 2. Force ratio score (high tangential force suggests scraping)
        force_ratio = features['force_ratio']
        force_score = min(1.0, force_ratio / max(self.force_ratio_threshold, 0.1))
        scores['force_score'] = force_score
        
        # 3. Contact angle score (shallow angles suggest scraping)
        angle = features['contact_angle']
        angle_score = 1.0 - min(1.0, angle / (np.pi/2))  # 0 at 90°, 1 at 0°
        scores['angle_score'] = angle_score
        
        # 4. Energy ratio score (low tangential energy suggests scraping)
        energy_ratio = features['energy_ratio']
        energy_score = 1.0 - min(1.0, np.log10(energy_ratio + 1) / 3.0)  # Log scale for wide range
        scores['energy_score'] = energy_score
        
        # Weighted combination
        weights = {
            'velocity_score': 0.35,
            'force_score': 0.35,
            'angle_score': 0.15,
            'energy_score': 0.15
        }
        
        weighted_score = sum(scores[key] * weights[key] for key in scores)
        
        # Apply hysteresis to prevent rapid switching
        if previous_classification == 'scraping':
            threshold_low = 0.4  # Lower threshold to switch from scraping
            threshold_high = 0.6  # Higher threshold to switch to scraping
        else:
            threshold_low = 0.3
            threshold_high = 0.5
        
        # Classification
        if weighted_score >= threshold_high:
            classification = 'scraping'
        elif weighted_score <= threshold_low:
            classification = 'sliding'
        else:
            # Maintain previous classification if in hysteresis zone
            classification = previous_classification if previous_classification else 'sliding'
        
        # Calculate confidence
        confidence = abs(weighted_score - 0.5) * 2  # 0 at 0.5, 1 at 0 or 1
        
        # Add metadata
        scores['weighted_score'] = weighted_score
        scores['confidence'] = confidence
        scores['velocity_threshold'] = self.velocity_threshold
        scores['force_ratio_threshold'] = self.force_ratio_threshold
        
        return classification, scores
