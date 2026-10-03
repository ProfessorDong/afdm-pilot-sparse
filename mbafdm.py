"""Sample-exact multi-block AFDM simulator.

Everything is generated at the sample level on one contiguous transmit stream
(blocks, each with its own prefix), passed through a physical delay-Doppler
channel, then cut back into blocks. Inter-block Doppler phase, prefix
truncation and pulse-shaping tails are therefore produced by the physics,
never inserted by hand.

Conventions
-----------
- DAFT  F = D_{c2}^* F_N D_{c1}^*  (unitary), IDAFT = F^H.
- c1 = Q/(2N), Q = 2(alpha_max + xi) + 1 odd, N even: the chirp-periodic prefix
  reduces to an ordinary cyclic prefix (Bemani et al., TWC 2023).
- Path p: delay tau_p (samples, fractional allowed), Doppler kappa_p (subcarrier
  spacings), optional Doppler rate rho_p (kappa units per block), gain h_p.
  Absolute sample n = 0 is the first prefix sample of block 0.
- A path lands in the chirp domain at index m0 + alpha - Q*ell (delay shifts the
  index DOWN by Q per sample, Doppler shifts it UP by one per subcarrier spacing).
- Power: every block carries total energy N (unit average energy per chirp).
  In a pilot block the pilot takes the energy of the null guard it sits in, so
  data symbols keep unit energy in both block types.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MBAFDM:
    N: int = 512
    Ncp: int = 8
    alpha_max: int = 3     # max integer Doppler index the chirp design separates
    xi: int = 2            # extra chirp guard for fractional-Doppler spreading
    ell_max: int = 7       # max (integer part of) delay in samples
    m0: int | None = None  # pilot chirp index; default N//2
    c2: float | None = None
    Q: int = field(init=False)
    c1: float = field(init=False)

    def __post_init__(self):
        assert self.N % 2 == 0
        assert self.Ncp >= self.ell_max
        self.Q = 2 * (self.alpha_max + self.xi) + 1
        self.c1 = self.Q / (2 * self.N)
        if self.c2 is None:
            self.c2 = 1.0 / (2 * self.N)
        if self.m0 is None:
            self.m0 = self.N // 2
        n = np.arange(self.N)
        self.dc1 = np.exp(1j * 2 * np.pi * self.c1 * n * n)
        self.dc2 = np.exp(1j * 2 * np.pi * self.c2 * n * n)
        self.beta = (self.N + self.Ncp) / self.N
        a = self.alpha_max + self.xi
        # pilot-response window W: offsets from m0 of every path the design admits
        self.win_off = np.arange(-(self.Q * self.ell_max + a), a + 1)
        self.W = (self.m0 + self.win_off) % self.N
        # null guard: no data chirp may leak into W through any admissible path
        g = self.Q * self.ell_max + 2 * a
        self.zero_set = (self.m0 + np.arange(-g, g + 1)) % self.N
        mask = np.ones(self.N, bool)
        mask[self.zero_set] = False
        self.data_idx = np.flatnonzero(mask)
        self.Ep = float(len(self.zero_set))   # pilot inherits the guard's energy

    # ---------------- design properties ----------------
    @property
    def separable(self) -> bool:
        """Bemani non-overlap condition (l_max+1)(2(alpha_max+xi)+1) <= N."""
        return (self.ell_max + 1) * self.Q <= self.N

    @property
    def overhead(self) -> float:
        """Fraction of a pilot block's chirps not carrying data."""
        return len(self.zero_set) / self.N

    # ---------------- transforms ----------------
    def idaft(self, x):
        return np.fft.ifft(x * self.dc2, axis=-1, norm="ortho") * self.dc1

    def daft(self, r):
        return np.fft.fft(r * np.conj(self.dc1), axis=-1, norm="ortho") * np.conj(self.dc2)

    def index_of(self, ell, kappa):
        return (self.m0 + int(round(kappa)) - self.Q * int(round(ell))) % self.N

    # ---------------- frames ----------------
    @staticmethod
    def qpsk(rng, shape):
        return np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, size=shape)))

    def pilot_block(self, rng, data=True):
        x = np.zeros(self.N, complex)
        if data:
            x[self.data_idx] = self.qpsk(rng, self.data_idx.size)
        x[self.m0] = np.sqrt(self.Ep)
        return x

    def data_block(self, rng):
        return self.qpsk(rng, self.N)

    def frame(self, kinds, rng):
        """kinds: sequence of 'P' (pilot+guard+data) or 'D' (data on all chirps)."""
        return np.stack([self.pilot_block(rng) if k == "P" else self.data_block(rng) for k in kinds])

    def transmit(self, x):
        s = self.idaft(x)
        return np.concatenate([s[:, -self.Ncp:], s], axis=1).reshape(-1)

    def receive(self, r, B):
        r = np.asarray(r)
        blk = r[..., : B * (self.N + self.Ncp)].reshape(r.shape[:-1] + (B, self.N + self.Ncp))[..., self.Ncp:]
        return self.daft(blk)

    # ---------------- physical channel ----------------
    def channel(self, stream, tau, kappa, h, rho=None, born=None, taps=24,
                cfo=0.0, pn_var=0.0, rng=None, n0=0):
        """Doubly-dispersive channel applied along the last axis of `stream`
        (leading axes are independent streams, e.g. probe batches).

        tau   : delays in samples (fractional -> band-limited interpolation with a
                Kaiser-windowed sinc of +-taps; integer -> exact shift)
        kappa : Doppler at block 0 (subcarrier spacings)
        rho   : Doppler rate (kappa units per block period), default 0
        born  : first block index in which each path exists, default 0
        cfo   : common frequency offset (subcarrier spacings), Rx side
        pn_var: Wiener phase-noise increment variance per sample (rad^2), Rx side
        """
        stream = np.asarray(stream)
        L = stream.shape[-1]
        n = n0 + np.arange(L)          # absolute sample index
        Tb = self.N + self.Ncp
        P = len(tau)
        rho = np.zeros(P) if rho is None else np.asarray(rho, float)
        born = np.zeros(P, int) if born is None else np.asarray(born, int)
        r = np.zeros(stream.shape, complex)

        def shift(x, d):
            out = np.zeros_like(x, dtype=complex)
            if d >= 0:
                out[..., d:] = x[..., : L - d]
            else:
                out[..., : L + d] = x[..., -d:]
            return out

        for p in range(P):
            t = float(tau[p])
            if abs(t - round(t)) < 1e-12:
                sh = shift(stream, int(round(t)))
            else:
                k = np.arange(-taps, taps + 1) + int(np.floor(t))
                w = np.kaiser(2 * taps + 1, 8.0)
                g = np.sinc(k - t) * w          # g[k] multiplies s[n - k]
                sh = np.zeros(stream.shape, complex)
                for kk, gg in zip(k, g):
                    sh += gg * shift(stream, int(kk))
            ph = 2 * np.pi * (kappa[p] * n + 0.5 * rho[p] * n * n / Tb) / self.N
            path = h[p] * sh * np.exp(1j * ph)
            if born[p] > 0:
                path[..., n < born[p] * Tb] = 0
            r += path
        if cfo:
            r = r * np.exp(1j * 2 * np.pi * cfo * n / self.N)
        if pn_var > 0:
            rng = rng or np.random.default_rng()
            r = r * np.exp(1j * np.cumsum(np.sqrt(pn_var) * rng.standard_normal(L)))
        return r

    # ---------------- pilot atoms ----------------
    def atoms(self, ells, kappas):
        """Block-0 response on W of a unit pilot through one path (integer ell).

        Shape (len(ells), len(kappas), |W|). Block b's response is exactly
        exp(j2pi kappa b beta) times this (prefix >= delay, 2Nc1 integer).
        """
        e = np.zeros(self.N, complex)
        e[self.m0] = 1.0
        s = self.idaft(e)
        n_abs = np.arange(self.Ncp, self.Ncp + self.N)
        kappas = np.atleast_1d(kappas)
        out = np.empty((len(ells), len(kappas), len(self.W)), complex)
        for i, l in enumerate(ells):
            seg = np.roll(s, int(l))
            ph = np.exp(1j * 2 * np.pi * np.outer(kappas, n_abs) / self.N)
            out[i] = self.daft(seg[None, :] * ph)[:, self.W]
        return out

    # ---------------- legacy helper used by proto/gate1-4 ----------------
    def make_frame(self, B, rng, Ep=None, data=True):
        x = np.zeros((B, self.N), complex)
        if data:
            x[:, self.data_idx] = self.qpsk(rng, (B, self.data_idx.size))
        x[:, self.m0] = np.sqrt(self.Ep if Ep is None else Ep)
        return x
