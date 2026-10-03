"""Rate-1/2 K=7 convolutional code (133,171 octal), BICM with QPSK, soft Viterbi.

One codeword per block: the block's data chirps carry 2*n_sym coded bits =
2*(n_info + 6) after zero-tail termination. A fixed random interleaver per
codeword length spreads coded bits over chirps.
"""
from __future__ import annotations

import numpy as np

K = 7
G = (0o133, 0o171)
NS = 1 << (K - 1)


def _tables():
    nxt = np.zeros((NS, 2), int)
    out = np.zeros((NS, 2, 2), int)
    for s in range(NS):
        for u in range(2):
            reg = (u << (K - 1)) | s                 # newest bit at MSB
            for j, g in enumerate(G):
                out[s, u, j] = bin(reg & g).count("1") & 1
            nxt[s, u] = reg >> 1
    return nxt, out


NXT, OUT = _tables()
# predecessor tables for vectorised Viterbi
PRED = np.zeros((NS, 2), int)
PIN = np.zeros((NS, 2), int)
_cnt = np.zeros(NS, int)
for s in range(NS):
    for u in range(2):
        t = NXT[s, u]
        PRED[t, _cnt[t]] = s
        PIN[t, _cnt[t]] = u
        _cnt[t] += 1


def n_info_for(n_sym):
    """Info bits carried by a block with n_sym QPSK symbols (2 coded bits each)."""
    return n_sym - (K - 1)


def encode(bits):
    bits = np.concatenate([bits, np.zeros(K - 1, int)])
    s = 0
    c = np.empty((bits.size, 2), int)
    for i, u in enumerate(bits):
        c[i] = OUT[s, u]
        s = NXT[s, u]
    return c.reshape(-1)


def viterbi(llr):
    """Max-log soft Viterbi. llr[i] = log P(c_i=0)/P(c_i=1). Returns info bits."""
    T = llr.size // 2
    L = llr.reshape(T, 2)
    sgn = 1 - 2 * OUT                                  # (NS, 2, 2): +1 for bit 0
    pm = np.full(NS, -np.inf); pm[0] = 0.0
    surv = np.zeros((T, NS), np.int8)
    # branch metric into state t from (PRED[t,j], PIN[t,j])
    for i in range(T):
        bm = 0.5 * np.sum(sgn[PRED, PIN] * L[i], axis=-1)   # (NS, 2)
        cand = pm[PRED] + bm
        j = np.argmax(cand, axis=1)
        pm = cand[np.arange(NS), j]
        surv[i] = j
    s = 0                                              # zero-terminated
    u = np.empty(T, int)
    for i in range(T - 1, -1, -1):
        j = surv[i, s]
        u[i] = PIN[s, j]
        s = PRED[s, j]
    return u[: T - (K - 1)]


CRC_POLY = 0x1021  # CRC-16-CCITT


def crc16(bits):
    reg = 0xFFFF
    for b in bits:
        top = ((reg >> 15) & 1) ^ int(b)
        reg = (reg << 1) & 0xFFFF
        if top:
            reg ^= CRC_POLY
    return np.array([(reg >> (15 - i)) & 1 for i in range(16)], int)


class BlockCode:
    """Per-block codec for a given number of QPSK symbols. The info word ends
    with a CRC-16 over the payload, so k_payload = k - 16."""

    _perm: dict = {}

    def __init__(self, n_sym):
        self.n_sym = n_sym
        self.k = n_info_for(n_sym)
        self.k_payload = self.k - 16
        if n_sym not in BlockCode._perm:
            BlockCode._perm[n_sym] = np.random.default_rng(12345 + n_sym).permutation(2 * n_sym)
        self.perm = BlockCode._perm[n_sym]

    def attach_crc(self, payload):
        return np.concatenate([payload, crc16(payload)])

    def check(self, info):
        return bool(np.array_equal(crc16(info[:-16]), info[-16:]))

    def modulate(self, info):
        c = encode(info)[self.perm]
        b = c.reshape(-1, 2)
        return ((1 - 2 * b[:, 0]) + 1j * (1 - 2 * b[:, 1])) / np.sqrt(2)

    def demod_decode(self, zu, v):
        llr = np.empty(2 * self.n_sym)
        llr[0::2] = 2 * np.sqrt(2) * zu.real / v
        llr[1::2] = 2 * np.sqrt(2) * zu.imag / v
        inv = np.empty_like(llr)
        inv[self.perm] = llr
        return viterbi(np.clip(inv, -50, 50))
