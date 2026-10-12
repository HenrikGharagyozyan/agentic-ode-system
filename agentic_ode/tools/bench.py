"""Simulated optical bench: laser -> DMD (binary amplitude) -> scattering medium -> camera.

The camera records I = |T E_in|^2 plus noise (Poisson shot noise and Gaussian read noise),
so two shots of one DMD pattern differ. T is fixed by a seed: every run sees the same medium.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

DMD_SIDE = 64
CAMERA_SIDE = 64
MACROPIXEL_SIZES = (2, 4, 8, 16)
MEDIUM_SEED = 2026
PHOTONS_PER_SPECKLE = 200.0  # mean counts of one camera pixel for a random half-on pattern
READ_NOISE = 2.0  # standard deviation of the Gaussian read noise, in counts
_SCALE = PHOTONS_PER_SPECKLE / (DMD_SIDE * DMD_SIDE / 2)  # |field|^2 -> expected counts


@lru_cache(maxsize=1)
def transmission_matrix() -> np.ndarray:
    """T[camera pixel, DMD pixel]: complex circular Gaussian with unit variance (128 MB, complex64)."""
    rng = np.random.default_rng(MEDIUM_SEED)
    shape = (CAMERA_SIDE * CAMERA_SIDE, DMD_SIDE * DMD_SIDE)
    t = np.empty(shape, dtype=np.complex64)
    t.real = rng.standard_normal(shape, dtype=np.float32)
    t.imag = rng.standard_normal(shape, dtype=np.float32)
    t /= np.sqrt(2.0)
    return t


@lru_cache(maxsize=len(MACROPIXEL_SIZES))
def macropixel_matrix(m: int) -> np.ndarray:
    """T with the columns of each m x m DMD block summed: shape (camera pixels, (64/m)^2)."""
    k = DMD_SIDE // m
    t = transmission_matrix().reshape(CAMERA_SIDE * CAMERA_SIDE, k, m, k, m)
    return np.ascontiguousarray(t.sum(axis=(2, 4)).reshape(CAMERA_SIDE * CAMERA_SIDE, k * k))


class Bench:
    """One DMD pattern of macropixels (True = mirror on) and the field it makes on the camera."""

    def __init__(self, macropixel: int, seed: int):
        if macropixel not in MACROPIXEL_SIZES:
            raise ValueError(f"macropixel must be one of {MACROPIXEL_SIZES}, got {macropixel}")
        self.m = macropixel
        self.side = DMD_SIDE // macropixel
        self.n = self.side * self.side  # number of controlled macropixels
        self.tm = macropixel_matrix(macropixel)
        self.rng = np.random.default_rng(seed)
        self.pattern = np.zeros(self.n, dtype=bool)
        self.field = np.zeros(CAMERA_SIDE * CAMERA_SIDE, dtype=np.complex64)

    def set_pattern(self, pattern: np.ndarray) -> None:
        self.pattern = np.asarray(pattern, dtype=bool).copy()
        self.field = self.tm @ self.pattern.astype(np.complex64)

    def flip(self, idx: np.ndarray) -> None:
        """Flip some macropixels. Only their columns change the field, so this is cheap."""
        sign = np.where(self.pattern[idx], -1.0, 1.0).astype(np.complex64)
        self.field += self.tm[:, idx] @ sign
        self.pattern[idx] = ~self.pattern[idx]

    def shot(self) -> np.ndarray:
        """One noisy camera image (counts), shape (64, 64)."""
        expected = (np.abs(self.field) ** 2) * _SCALE
        counts = self.rng.poisson(expected).astype(np.float64)
        counts += self.rng.normal(0.0, READ_NOISE, counts.shape)
        return counts.reshape(CAMERA_SIDE, CAMERA_SIDE)

    def dmd_image(self) -> np.ndarray:
        """The pattern at DMD pixel resolution (64 x 64, 0/1)."""
        return np.kron(self.pattern.reshape(self.side, self.side), np.ones((self.m, self.m))).astype(np.uint8)


@dataclass(frozen=True)
class Regions:
    """Focus square and the proximal background ring, split into sectors (masks on the camera)."""

    focus: np.ndarray
    sectors: tuple[np.ndarray, ...]
    centre: tuple[float, float]
    inner: float  # ring inner half-width (Chebyshev distance from the focus centre)
    outer: float


def make_regions(focus_size: int, n_sectors: int, centre: tuple[int, int] | None = None,
                 gap: int = 2, ring_width: int = 6) -> Regions:
    cy, cx = centre or (CAMERA_SIDE // 2, CAMERA_SIDE // 2)
    y0, x0 = cy - focus_size // 2, cx - focus_size // 2
    yy, xx = np.mgrid[0:CAMERA_SIDE, 0:CAMERA_SIDE]
    focus = (yy >= y0) & (yy < y0 + focus_size) & (xx >= x0) & (xx < x0 + focus_size)
    fy, fx = y0 + (focus_size - 1) / 2, x0 + (focus_size - 1) / 2
    dist = np.maximum(np.abs(yy - fy), np.abs(xx - fx))
    inner = (focus_size - 1) / 2 + gap
    outer = inner + ring_width
    ring = (dist > inner) & (dist <= outer)
    angle = np.arctan2(yy - fy, xx - fx)
    sector = np.floor((angle + np.pi) / (2 * np.pi) * n_sectors).astype(int) % n_sectors
    sectors = tuple(ring & (sector == k) for k in range(n_sectors))
    return Regions(focus, sectors, (fy, fx), inner, outer)


def enhancement(image: np.ndarray, regions: Regions) -> float:
    """Mean intensity in the focus divided by the mean of the sector means of the background ring."""
    background = np.mean([image[s].mean() for s in regions.sectors])
    return float(image[regions.focus].mean() / max(background, 1e-9))
