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
from scipy.interpolate import CubicSpline
from typing import Optional, Union, Tuple

class CubicSplineWithNaN:
    """
    Cubic spline interpolation that preserves NaN values in y-coordinates.
    
    This class splits the data at NaN positions, fits separate cubic splines
    to each continuous segment, and preserves NaN values in the output.
    
    Parameters
    ----------
    x : array_like
        1-D array of x-coordinates. Must be strictly increasing.
    y : array_like
        1-D array of y-coordinates. May contain NaN values.
    bc_type : str or tuple, optional
        Boundary condition type. Options are:
        - 'not-a-knot' (default)
        - 'periodic'
        - 'clamped' (tuple with first derivatives at endpoints)
        - 'natural' (tuple with second derivatives at endpoints)
    extrapolate : bool or 'periodic', optional
        If bool, determines whether to extrapolate beyond the base interval.
        If 'periodic', periodic extrapolation is used.
    axis : int, optional
        Axis along which y is assumed to be varying.
    
    Attributes
    ----------
    segments : list
        List of tuples (x_segment, y_segment, spline) for each continuous segment
    nan_mask : ndarray
        Boolean mask indicating NaN positions in the original y array
    x_min : float
        Minimum x value in the data
    x_max : float
        Maximum x value in the data
    """
    
    def __init__(self, x: np.ndarray, y: np.ndarray, bc_type: Union[str, Tuple] = 'not-a-knot', extrapolate: Union[bool, str] = True, axis: int = 0):
        
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.bc_type = bc_type
        self.extrapolate = extrapolate
        self.axis = axis
        
        # Validate inputs
        if self.x.ndim != 1:
            raise ValueError("x must be 1-dimensional")
        if self.x.shape[0] != self.y.shape[axis]:
            raise ValueError(f"x and y must have same length along axis {axis}")
        
        # Find NaN positions
        self.nan_mask = np.isnan(self.y)
        
        # Split data into continuous segments (non-NaN regions)
        self.segments = self._split_into_segments()
        
        # Store min and max x for bounds checking
        self.x_min = np.min(self.x)
        self.x_max = np.max(self.x)
    
    def _split_into_segments(self) -> list:
        """Split data into continuous segments separated by NaN values."""
        segments = []
        
        # Find indices where segments start and end
        nan_indices = np.where(self.nan_mask)[0]
        
        if len(nan_indices) == 0:
            # No NaN values - single segment
            if len(self.x) >= 2:
                spline = CubicSpline(self.x, self.y, 
                                    bc_type=self.bc_type,
                                    extrapolate=self.extrapolate,
                                    axis=self.axis)
                segments.append((self.x, self.y, spline))
        else:
            # Find continuous segments between NaN values
            start_idx = 0
            
            for nan_idx in nan_indices:
                if start_idx < nan_idx:
                    # Extract segment before NaN
                    segment_x = self.x[start_idx:nan_idx]
                    segment_y = np.take(self.y, range(start_idx, nan_idx), axis=self.axis)
                    
                    # Only create spline if segment has at least 2 points
                    if len(segment_x) >= 2:
                        spline = CubicSpline(segment_x, segment_y,
                                            bc_type=self.bc_type,
                                            extrapolate=self.extrapolate,
                                            axis=self.axis)
                        segments.append((segment_x, segment_y, spline))
                
                start_idx = nan_idx + 1
            
            # Check for segment after last NaN
            if start_idx < len(self.x):
                segment_x = self.x[start_idx:]
                segment_y = np.take(self.y, range(start_idx, len(self.x)), axis=self.axis)
                
                if len(segment_x) >= 2:
                    spline = CubicSpline(segment_x, segment_y,
                                        bc_type=self.bc_type,
                                        extrapolate=self.extrapolate,
                                        axis=self.axis)
                    segments.append((segment_x, segment_y, spline))
        
        return segments
    
    def __call__(self, x_new: np.ndarray) -> np.ndarray:
        """
        Evaluate the spline at new points.
        
        Parameters
        ----------
        x_new : array_like
            Points at which to evaluate the spline.
            
        Returns
        -------
        y_new : ndarray
            Interpolated values. NaN values are preserved at positions
            corresponding to original NaN y-values or outside segments.
        """
        x_new = np.asarray(x_new, dtype=float)
        original_shape = x_new.shape
        
        # Flatten for processing
        x_flat = x_new.flatten()
        y_flat = np.full(x_flat.shape, np.nan, dtype=float)
        
        # For each segment, interpolate values within its x-range
        for segment_x, segment_y, spline in self.segments:
            # Find indices where x_new falls within this this segment's range
            segment_min = segment_x[0]
            segment_max = segment_x[-1]
            
            # Include a small tolerance for floating point comparisons
            mask = (x_flat >= segment_min - 1e-12) & (x_flat <= segment_max + 1e-12)
            
            if np.any(mask):
                # Evaluate spline for points in this segment
                y_flat[mask] = spline(x_flat[mask])
        
        # Handle extrapolation if requested
        if self.extrapolate and len(self.segments) > 0:
            # For points outside all segments, use nearest segment's spline
            # with extrapolation
            below_mask = x_flat < self.x_min
            above_mask = x_flat > self.x_max
            
            if np.any(below_mask) and len(self.segments) > 0:
                # Use first segment for extrapolation to the left
                y_flat[below_mask] = self.segments[0][2](x_flat[below_mask])
            
            if np.any(above_mask) and len(self.segments) > 0:
                # Use last segment for extrapolation to the right
                y_flat[above_mask] = self.segments[-1][2](x_flat[above_mask])
        
        # Reshape to original input shape
        return y_flat.reshape(original_shape)
    
    def derivative(self, nu: int = 1) -> 'CubicSplineWithNaN':
        """
        Return a CubicSplineWithNaN representing the derivative.
        
        Parameters
        ----------
        nu : int, optional
            Order of derivative to compute. Default is 1.
            
        Returns
        -------
        spline_deriv : CubicSplineWithNaN
            A new CubicSplineWithNaN object representing the derivative.
        """
        # Create a new object with derivative values
        y_deriv = np.full_like(self.y, np.nan)
        
        for segment_x, segment_y, spline in self.segments:
            # Get indices for this segment in the original arrays
            start_idx = np.where(self.x == segment_x[0])[0][0]
            end_idx = np.where(self.x == segment_x[-1])[0][0] + 1
            
            # Compute derivative at original x points
            y_deriv[start_idx:end_idx] = spline(segment_x, nu)
        
        return CubicSplineWithNaN(self.x, y_deriv, 
                                 bc_type=self.bc_type,
                                 extrapolate=self.extrapolate,
                                 axis=self.axis)
    
    def antiderivative(self, nu: int = 1) -> 'CubicSplineWithNaN':
        """
        Return a CubicSplineWithNaN representing the antiderivative.
        
        Parameters
        ----------
        nu : int, optional
            Order of antiderivative to compute. Default is 1.
            
        Returns
        -------
        spline_anti : CubicSplineWithNaN
            A new CubicSplineWithNaN object representing the antiderivative.
        """
        y_anti = np.full_like(self.y, np.nan)
        
        for segment_x, segment_y, spline in self.segments:
            # Get indices for this segment in the original arrays
            start_idx = np.where(self.x == segment_x[0])[0][0]
            end_idx = np.where(self.x == segment_x[-1])[0][0] + 1
            
            # Compute antiderivative at original x points
            spline_anti = spline.antiderivative(nu)
            y_anti[start_idx:end_idx] = spline_anti(segment_x)
        
        return CubicSplineWithNaN(self.x, y_anti,
                                 bc_type=self.bc_type,
                                 extrapolate=self.extrapolate,
                                 axis=self.axis)
    
    @property
    def c(self) -> np.ndarray:
        """Coefficients of the spline pieces."""
        # Return coefficients for all segments concatenated with NaN padding
        total_points = len(self.x)
        coeff_shape = (total_points - 1, 4) if total_points > 1 else (0, 4)
        coeffs = np.full(coeff_shape, np.nan)
        
        for segment_x, _, spline in self.segments:
            # Find where this segment fits in the overall coefficient array
            start_idx = np.where(self.x == segment_x[0])[0][0]
            segment_len = len(segment_x)
            
            if segment_len >= 2:
                coeffs[start_idx:start_idx + segment_len - 1] = spline.c.T
        
        return coeffs.T
    
    def get_segment_info(self) -> list:
        """
        Get information about each continuous segment.
        
        Returns
        -------
        info : list of dict
            List containing dictionaries with segment information:
            - 'x_range': (min_x, max_x)
            - 'length': number of points in segment
            - 'has_spline': whether a spline was fitted (requires ≥2 points)
        """
        info = []
        for segment_x, segment_y, spline in self.segments:
            info.append({
                'x_range': (segment_x[0], segment_x[-1]),
                'length': len(segment_x),
                'has_spline': True
            })
        return info
