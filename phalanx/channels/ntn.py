"""Non-terrestrial network (NTN) orbital channel model.

Models LEO satellite links with:
- Orbital geometry (altitude, inclination, elevation-dependent visibility)
- Free-space path loss with atmosphere absorption
- Doppler shift from satellite motion
- Rice fading with elevation-dependent K-factor
- Per-link SNR observation

This is a domain-specific channel for NTN/6G multi-link evaluation.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from phalanx.config import ChannelConfig
from phalanx.core import Channel

_EARTH_RADIUS_KM = 6371.0
_SPEED_OF_LIGHT = 3e8  # m/s
_BOLTZMANN = 1.38e-23  # J/K


class NTNOrbitalChannel(Channel):
    """LEO satellite channel with orbital dynamics.

    Produces per-link observations comprising (SNR_dB, doppler_hz,
    elevation_deg, path_loss_dB) -- so ``d_obs = 4``.

    Args:
        cfg: Channel configuration (uses NTN-specific fields).
        rng: NumPy random generator.
        ground_positions_km: ``(M, 2)`` array of ground terminal
            (lat, lon) in km-scale planar coords.  If *None*, terminals
            are placed uniformly in a 100 km x 100 km area.
    """

    def __init__(
        self,
        cfg: ChannelConfig,
        rng: np.random.Generator,
        ground_positions_km: Optional[np.ndarray] = None,
    ):
        self.cfg = cfg
        self.rng = rng
        self.M = cfg.M
        self.d_obs = 4  # (SNR_dB, doppler_hz, elevation_deg, path_loss_dB)

        if ground_positions_km is None:
            self.ground_pos = rng.uniform(0, 100, size=(cfg.M, 2))
        else:
            self.ground_pos = ground_positions_km.copy()

        self.altitude_km = cfg.altitude_km
        self.inclination_rad = np.deg2rad(cfg.inclination_deg)
        self.carrier_freq_hz = cfg.carrier_freq_ghz * 1e9
        self.elevation_min_rad = np.deg2rad(cfg.elevation_min_deg)

        # Orbital period (Kepler)
        orbit_radius = _EARTH_RADIUS_KM + self.altitude_km
        self.orbital_period_s = (
            2 * np.pi * np.sqrt((orbit_radius * 1e3) ** 3 / 3.986e14)
        )
        self.angular_velocity = 2 * np.pi / self.orbital_period_s  # rad/s

        # Satellite sub-point trajectory state (angle along orbit)
        self.theta = rng.uniform(0, 2 * np.pi)  # initial orbital angle
        self.t = 0

    # -- Orbital geometry helpers --

    def _satellite_position_km(self) -> np.ndarray:
        """Compute satellite ground-track position in the planar coord frame."""
        r = _EARTH_RADIUS_KM + self.altitude_km
        x = r * np.cos(self.theta) * np.cos(self.inclination_rad)
        y = r * np.sin(self.theta)
        # Project to 2-D ground coords (simplified planar)
        return np.array([50.0 + x * 0.01, 50.0 + y * 0.01])

    def _elevation_angle(self, ground_pos_km: np.ndarray, sat_pos_km: np.ndarray) -> float:
        """Compute elevation angle from ground terminal to satellite (degrees)."""
        dx = sat_pos_km - ground_pos_km
        horiz_dist_km = np.linalg.norm(dx)
        if horiz_dist_km < 1e-6:
            return 90.0
        elev_rad = np.arctan2(self.altitude_km, horiz_dist_km)
        return np.rad2deg(elev_rad)

    def _slant_range_km(self, elev_deg: float) -> float:
        """Compute slant range from elevation angle."""
        elev_rad = np.deg2rad(max(elev_deg, 1.0))
        return self.altitude_km / np.sin(elev_rad)

    def _free_space_path_loss_db(self, slant_range_km: float) -> float:
        """FSPL in dB: 20 log10(4 pi d f / c)."""
        d_m = slant_range_km * 1e3
        fspl = 20 * np.log10(4 * np.pi * d_m * self.carrier_freq_hz / _SPEED_OF_LIGHT)
        return fspl

    def _doppler_hz(self, elev_deg: float) -> float:
        """Doppler shift at current elevation."""
        v_sat = self.angular_velocity * (_EARTH_RADIUS_KM + self.altitude_km) * 1e3  # m/s
        elev_rad = np.deg2rad(max(elev_deg, 1.0))
        # Doppler = v_sat * cos(elev) * f_c / c
        return v_sat * np.cos(elev_rad) * self.carrier_freq_hz / _SPEED_OF_LIGHT

    def _rice_fading_db(self, elev_deg: float) -> float:
        """Rice fading with elevation-dependent K-factor.

        Higher elevation -> stronger LoS -> higher K -> less fading.
        """
        K = 2.0 + 0.2 * max(elev_deg, 0.0)  # K-factor increases with elevation
        # Rice fading envelope magnitude (Nakagami-m approximation)
        m = (K + 1) ** 2 / (2 * K + 1)
        fading_linear = self.rng.gamma(m, 1.0 / m)
        return 10.0 * np.log10(max(fading_linear, 1e-10))

    def step(self) -> np.ndarray:
        """Advance orbit by one slot and return per-link observations.

        Returns:
            Array of shape ``(M, 4)`` with columns
            (SNR_dB, doppler_hz, elevation_deg, path_loss_dB).
        """
        # Advance orbital angle (1 second per slot by default)
        self.theta += self.angular_velocity
        self.theta %= 2 * np.pi
        self.t += 1

        sat_pos = self._satellite_position_km()
        obs = np.zeros((self.M, 4))

        for m in range(self.M):
            elev = self._elevation_angle(self.ground_pos[m], sat_pos)

            if elev < np.rad2deg(self.elevation_min_rad):
                # Below minimum elevation -- link unavailable
                obs[m] = [0.0, 0.0, elev, 200.0]
                continue

            slant = self._slant_range_km(elev)
            fspl = self._free_space_path_loss_db(slant)
            fading = self._rice_fading_db(elev)
            atm_loss = 0.5 + 2.0 / max(np.sin(np.deg2rad(elev)), 0.1)  # atmosphere
            total_pl = fspl + atm_loss - fading

            # SNR = P_tx - PL - N (simplified link budget)
            p_tx_dbm = 30.0  # 1 W
            noise_dbm = -174.0 + 10 * np.log10(20e6)  # 20 MHz BW
            snr_db = p_tx_dbm - total_pl - noise_dbm

            doppler = self._doppler_hz(elev)

            obs[m] = [snr_db, doppler, elev, total_pl]

        return obs

    def get_hidden_state(self) -> np.ndarray:
        """Return orbital angle as hidden state."""
        return np.array([self.theta])

    def save_state(self) -> Dict[str, Any]:
        return {
            "theta": float(self.theta),
            "t": self.t,
            "rng_state": self.rng.bit_generator.state,
        }

    def restore_state(self, saved: Dict[str, Any]) -> None:
        self.theta = saved["theta"]
        self.t = saved["t"]
        self.rng.bit_generator.state = saved["rng_state"]

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        if rng is not None:
            self.rng = rng
        self.theta = self.rng.uniform(0, 2 * np.pi)
        self.t = 0
