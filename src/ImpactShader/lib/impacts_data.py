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

import os, sys
import numpy as np
import math
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

class ContactType(Enum):
    """Types of contact interactions based on the paper"""
    IMPACT = "impact"          # Impulsive contact (collision)
    SCRAPING = "scraping"      # Scraping/sliding contact
    ROLLING = "rolling"        # Rolling contact
    MIXED = "mixed"           # Combination of contact types

@dataclass
class ContactForce:
    """Represents a contact force with temporal characteristics"""
    id_vertex: int                    # Vertex ID on the object
    force_vector: np.ndarray          # Force vector in world coordinates (N)
    contact_type: ContactType         # Type of contact
    contact_normal: np.ndarray        # Surface normal at contact point
    relative_velocity: np.ndarray     # Relative velocity at contact point (m/s)
    contact_area: float = 0.0         # Estimated contact area (m²)
    friction_coefficient: float = 0.3 # Dynamic friction coefficient
    rolling_radius: float = 0.0       # For rolling contacts, estimated radius
    duration: float = 0.0             # Expected contact duration (s)
    
    def __post_init__(self):
        # Ensure numpy arrays
        self.force_vector = np.array(self.force_vector, dtype=np.float32)
        self.contact_normal = np.array(self.contact_normal, dtype=np.float32)
        self.relative_velocity = np.array(self.relative_velocity, dtype=np.float32)
        
        # Normalize contact normal
        norm = np.linalg.norm(self.contact_normal)
        if norm > 0:
            self.contact_normal = self.contact_normal / norm

@dataclass
class ObjectContact:
    """Represents all contact interactions for a single object"""
    obj_idx: int                      # Object index
    contacts: List[ContactForce] = field(default_factory=list)  # List of contact forces
    convex_hull: Optional[np.ndarray] = None  # Convex hull vertices for smooth surfaces
    surface_roughness: float = 0.001  # Surface roughness parameter (m)
    material_hardness: float = 1.0    # Material hardness factor
    
    def add_contact(self, contact: ContactForce):
        """Add a contact force to this object"""
        self.contacts.append(contact)
    
    def get_contact_types(self) -> List[ContactType]:
        """Get unique contact types present in this object's contacts"""
        return list(set([c.contact_type for c in self.contacts]))
    
    def get_total_force(self) -> np.ndarray:
        """Calculate total force vector from all contacts"""
        if not self.contacts:
            return np.zeros(3)
        return sum((c.force_vector for c in self.contacts), np.zeros(3))

@dataclass
class ImpactEvent:
    """Represents a complete impact/contact event involving multiple objects"""
    idx: int                          # Event index
    start_time: float                 # Start time of the event (s)
    end_time: float                   # End time of the event (s)
    duration: float                   # Total duration (s)
    coord: Tuple[float, float, float] # Approximate event location
    object_contacts: List[ObjectContact] = field(default_factory=list)  # Contacts per object
    dominant_contact_type: ContactType = ContactType.IMPACT  # Dominant contact type
    
    # Audio generation parameters
    audio_duration: float = 3.0       # Duration of generated audio (s)
    sample_rate: int = 48000          # Audio sample rate
    
    def __post_init__(self):
        # Calculate duration if not provided
        if self.duration <= 0:
            self.duration = self.end_time - self.start_time
        
        # Determine dominant contact type
        self._determine_dominant_type()
    
    def add_object_contact(self, obj_contact: ObjectContact):
        """Add object contact data to this event"""
        self.object_contacts.append(obj_contact)
    
    def get_object_contact(self, obj_idx: int) -> Optional[ObjectContact]:
        """Get contact data for a specific object"""
        for obj_contact in self.object_contacts:
            if obj_contact.obj_idx == obj_idx:
                return obj_contact
        return None
    
    def get_all_vertices(self, obj_idx: int) -> List[int]:
        """Get all vertex IDs involved in contacts for a specific object"""
        obj_contact = self.get_object_contact(obj_idx)
        if obj_contact is None:
            return []
        return [c.id_vertex for c in obj_contact.contacts]
    
    def get_contact_audio_duration(self) -> float:
        """Get appropriate audio duration based on contact type"""
        if self.dominant_contact_type == ContactType.IMPACT:
            return min(self.audio_duration, 1.0)  # Short for impacts
        elif self.dominant_contact_type == ContactType.SCRAPING:
            return min(self.audio_duration, self.duration + 0.5)  # Follows contact duration
        elif self.dominant_contact_type == ContactType.ROLLING:
            return min(self.audio_duration, self.duration + 1.0)  # # Can be longer
        else:
            return self.audio_duration
    
    def _determine_dominant_type(self):
        """Determine the dominant contact type from all object contacts"""
        if not self.object_contacts:
            return
        
        # Count contact types
        type_counts = {}
        for obj_contact in self.object_contacts:
            for contact in obj_contact.contacts:
                type_name = contact.contact_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        if not type_counts:
            return
        
        # Find most frequent type
        dominant_type_name = max(type_counts, key=type_counts.get)
        
        # Set dominant type
        if dominant_type_name == "scraping":
            self.dominant_contact_type = ContactType.SCRAPING
        elif dominant_type_name == "rolling":
            self.dominant_contact_type = ContactType.ROLLING
        elif dominant_type_name == "impact":
            self.dominant_contact_type = ContactType.IMPACT
        else:
            self.dominant_contact_type = ContactType.MIXED
    
    def get_force_magnitude(self) -> float:
        """Calculate maximum force magnitude across all contacts"""
        max_force = 0.0
        for obj_contact in self.object_contacts:
            for contact in obj_contact.contacts:
                force_mag = np.linalg.norm(contact.force_vector)
                max_force = max(max_force, force_mag)
        return max_force
    
    def generate_contact_description(self) -> str:
        """Generate a human-readable description of the contact event"""
        desc = f"ImpactEvent {self.idx} at t={self.start_time:.3f}s"
        desc += f"  Location: {self.coord}\n"
        desc += f"  Duration: {self.duration:.3f}s"
        desc += f"  Dominant type: {self.dominant_contact_type.value}\n"
        desc += f"  Objects involved: {[oc.obj_idx for oc in self.object_contacts]}\n"
        
        for obj_contact in self.object_contacts:
            desc += f"  Object {obj_contact.obj_idx}:\n"
            for contact in obj_contact.contacts:
                force_mag = np.linalg.norm(contact.force_vector)
                vel_mag = np.linalg.norm(contact.relative_velocity)
                desc += f"    - Vertex {contact.id_vertex}: {contact.contact_type.value}, "
                desc += f"Force: {force_mag:.2f}N, Vel: {vel_mag:.2f}m/s"
        
        return desc
