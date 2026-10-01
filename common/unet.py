"""common/unet.py — the time-conditioned U-Net (artifact: `unet-skeleton`).

Built in U0.L4. **The interface below is FROZEN.** U3.L2 trains this network,
U3.L3 conditions it, PS0 imports it, and none of them may patch it — a change
after the U0.L4 freeze is a breaking change and needs a dated CHANGELOG note.

THE CONTRACT (U0.L4 notes §A; L3-U0-L4 §2A):

    UNet(in_ch, base_ch, ch_mults, attn_resolutions, num_classes=None)
    forward(x: (B,C,H,W) float,
            t: (B,)     float in [0,1],
            y: (B,)     long  optional) -> (B,C,H,W)

Three laws are carried by that signature, and each is enforced by an assert
in `forward` rather than by a docstring nobody reads:

  1. `t` is ALWAYS a (B,) tensor of floats in [0,1] — the FM time arrow at
     code level (t=0 noise -> t=1 data). Any later code that wants a
     different time variable converts OUTSIDE this network. This single line
     is what keeps the diffusion dialect out of the course codebase.
  2. `y` is optional and its pathway exists only when `num_classes` is set.
     The label embedding is SUMMED into the time embedding (U0.T4 @eq-embsum),
     never concatenated, so the modulation path does not grow with the number
     of conditioning signals.
  3. The output has the input's shape. It is a per-pixel quantity.

The class-conditional pathway is built today and deliberately unused. U3.L3
needs no architectural surgery — only classifier-free-guidance logic, which
is NOT here (no null token, no label dropout: forbidden content for U0.L4).

INTERIOR LAWS (the U0.T2 recipe card, unchanged; U0.T4 @sec-unet):
    GroupNorm (never BatchNorm — a generative model is sampled one at a
    time, so batch statistics would make one sample depend on the others),
    SiLU, and zero-initialization of every residual branch's last
    convolution plus every FiLM projection. At initialization this network
    is exactly the zero function, and it ignores t. Both are asserted by the
    U0.L4 Part C rituals.

EXTENSION DISCIPLINE (the U0.L2 lesson, applied to a second artifact): the
five positional arguments above are the frozen contract; everything added
later is keyword-only with a default, so every call written today keeps
working. `num_res_blocks`, `d_emb` and `dropout` are already in that shape.

In the student template the regions between SOLUTION markers are stripped to
TODO stubs (TODO 1, 2, 3 of the U0.L4 notebook); this file is the course's
reference implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["UNet", "ResBlock", "AttnBlock", "timestep_embedding", "num_groups"]


# ===========================================================================
# Provided helpers — read them, you are asked about one of them
# ===========================================================================


def num_groups(channels: int, maximum: int = 32) -> int:
    """Largest group count <= `maximum` that divides `channels`.

    GroupNorm needs `channels % num_groups == 0`. Hard-coding 32 is the
    reference-implementation habit and it crashes the moment a channel count
    is not a multiple of 32 — which happens as soon as you pick your own
    `base_ch`. The checkpoint question of Part B asks what this returns for
    the widths in your config, and why the answer is not always 32.
    """
    for g in (maximum, 16, 8, 4, 2, 1):
        if g <= channels and channels % g == 0:
            return g
    return 1  # unreachable for channels >= 1, kept so the function is total


def _norm(channels: int) -> nn.GroupNorm:
    """The course's normalization layer, in one place."""
    return nn.GroupNorm(num_groups(channels), channels)


SMALL_INIT_STD = 1e-3
"""Weight scale for the layers that are SMALL at init rather than ZERO.

L3-U0-L4 section 2A names exactly one exact-zero interior law: `conv_out`.
That one carries the whole "untrained field is the zero field" property, and
Part C ritual 1 asserts it bit for bit.

Zeroing anything else as well is what breaks the other three rituals, and it
was measured. With `conv_out` zero, ONE backward pass gives a nonzero
gradient to exactly two tensors in the whole network, `conv_out.weight` and
`conv_out.bias`, because every upstream path is multiplied by that zero
matrix. So:

  * a zero FiLM projection makes gamma(t) = beta(t) = 0 for every t, and the
    WIRING rituals 2 and 3 measure nothing at initialization;
  * a zero residual-branch `conv2` makes every ResBlock the identity, so the
    features reaching `conv_out` do not depend on t, and ritual 4 reports a
    difference of exactly 0.0 after its one prescribed SGD step.

Both are the vacuous-check failure the U0.L3 gradient ritual hit. Small
nonzero weights remove the cause while leaving the field exactly zero.
"""


def _zero_module(module: nn.Module) -> nn.Module:
    """Zero every parameter of `module` and return it (recipe-card law).

    Used on the last convolution of each residual branch, on each FiLM
    projection, and on the network's output convolution. The effect is that
    an untrained block is the identity and an untrained network is the zero
    field, which is the state the Part C rituals assert.
    """
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ResBlock(nn.Module):
    """PROVIDED COMPLETE. GroupNorm/SiLU residual block with FiLM (AdaGN).

    The chain is the U0.T2 recipe card; the modulation is U0.T4 @eq-film,
    applied immediately AFTER the second normalization — that is the point of
    the mechanism, because normalization has just removed the activation's
    scale and offset and modulation is what puts a conditioning-dependent
    scale and offset back:

        h <- (1 + gamma(c)) * norm(h) + beta(c)

    Writing `1 + gamma` rather than `gamma` means gamma = 0 is "no
    modulation"; with the projection zero-initialized, the block begins life
    ignoring its conditioning signal and learns to use it.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        d_emb: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = _norm(out_ch)
        self.dropout = nn.Dropout(dropout)
        # SMALL, not zero (see SMALL_INIT_STD): the branch starts nearly
        # silent, so the block is nearly the identity, without making the
        # features reaching the output head exactly t-independent.
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        nn.init.normal_(self.conv2.weight, std=SMALL_INIT_STD)
        nn.init.zeros_(self.conv2.bias)
        # One (gamma, beta) pair per channel, produced from the conditioning
        # vector c. Two numbers per channel per block, against the many
        # thousands in a convolution kernel: modulation is cheap.
        #
        # SMALL NONZERO, not zero (L3-U0-L4 section 2A). The zero-field law
        # is carried by `conv_out` alone; zeroing the FiLM projection as well
        # would make gamma(t) = beta(t) = 0 for EVERY t at initialization,
        # and the wiring rituals of Part C would then measure nothing — the
        # same vacuous-check defect the U0.L3 gradient ritual hit. With a
        # small nonzero projection the modulation already varies with t at
        # step 0, while the network's OUTPUT is still exactly the zero field.
        self.film = nn.Linear(d_emb, 2 * out_ch)
        nn.init.normal_(self.film.weight, std=SMALL_INIT_STD)
        nn.init.zeros_(self.film.bias)
        self.skip = (
            nn.Conv2d(in_ch, out_ch, kernel_size=1)
            if in_ch != out_ch
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.norm2(h)
        gamma, beta = self.film(F.silu(c)).chunk(2, dim=1)
        h = (1 + gamma[:, :, None, None]) * h + beta[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(h)))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    """PROVIDED COMPLETE. Single-head self-attention over the pixel grid.

    Cost is quadratic in the number of positions (U0.T4 @sec-attention), so
    a convolutional generator affords this at the bottom of its resolution
    ladder and nowhere else. The output projection is zero-initialized, so
    at initialization the block is the identity.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = _norm(channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.proj = _zero_module(nn.Conv2d(channels, channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, 3, c, h * w).unbind(dim=1)
        # (B, HW, C) x (B, C, HW) -> (B, HW, HW); scaling by 1/sqrt(C) keeps
        # the logits' scale independent of the channel count.
        attn = torch.softmax(
            torch.bmm(q.transpose(1, 2), k) / math.sqrt(c), dim=-1
        )
        out = torch.bmm(v, attn.transpose(1, 2)).reshape(b, c, h, w)
        return x + self.proj(out)


class Downsample(nn.Module):
    """PROVIDED COMPLETE. Halve the spatial resolution with a strided conv."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    """PROVIDED COMPLETE. Double the resolution, then smooth with a conv.

    Nearest-neighbour interpolation followed by a convolution, rather than a
    transposed convolution: the transposed convolution's uneven kernel
    overlap prints a checkerboard on generated images.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


# ===========================================================================
# TODO 1 — the sinusoidal time embedding (course spec, frozen at U0.T4)
# ===========================================================================


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """The course ladder. `t` is (B,) float in [0,1]; returns (B, dim).

    Frozen spec (U0.T4 @eq-ladder; L2-U0 sync note 9). With half = dim // 2:

        freqs = exp(-log(10000) * arange(half) / half)   # omega_k
        arg   = (1000 * t)[:, None] * freqs[None, :]
        emb   = cat([sin(arg), cos(arg)], dim=-1)

    The factor 1000 is the ONLY rescale, and it is deliberate: reference
    implementations were calibrated to a discrete schedule of about a
    thousand steps, so multiplying by 1000 here — and nowhere else — makes
    this from-scratch ladder produce the same numbers as DiT/ADM. A later
    cross-check against a library then reports real bugs instead of
    convention differences. Outside this function t is a float in [0,1]
    everywhere, and the FM arrow is untouched.
    """
    if dim % 2 != 0:
        raise ValueError(f"embedding width must be even, got {dim}")
    # --- SOLUTION: TODO 1 (stripped to a stub in the student notebook) ---
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=t.dtype, device=t.device)
        / half
    )
    arg = (1000.0 * t)[:, None] * freqs[None, :]
    return torch.cat([torch.sin(arg), torch.cos(arg)], dim=-1)
    # --- END SOLUTION (TODO 1) ---


# ===========================================================================
# The network
# ===========================================================================


class UNet(nn.Module):
    """The time-conditioned U-Net. Interface frozen at U0.L4.

    Args:
        in_ch: image channels (1 for MNIST, 3 for CIFAR).
        base_ch: channel width at full resolution.
        ch_mults: one multiplier per resolution level, coarsest last.
            `(1, 2, 2)` on 28x28 gives levels at 28, 14 and 7.
        attn_resolutions: spatial sizes at which to insert `AttnBlock`,
            e.g. `(7,)`. Sizes not reached by the ladder are ignored.
        num_classes: size of the label table, or None for no class pathway.

    Keyword-only extensions (added after the freeze, in the discipline of
    `common/train.py` v2 — every frozen call site keeps working):
        num_res_blocks: residual blocks per level (default 2).
        image_size: full-resolution side length, needed to turn
            `attn_resolutions` (written in pixels) into level indices at
            construction time (default 28, the MNIST side).
        d_emb: conditioning width (default `4 * base_ch`).
        dropout: dropout inside each residual branch (default 0).
    """

    def __init__(
        self,
        in_ch: int,
        base_ch: int,
        ch_mults: tuple[int, ...],
        attn_resolutions: tuple[int, ...],
        num_classes: int | None = None,
        *,
        num_res_blocks: int = 2,
        image_size: int = 28,
        d_emb: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.ch_mults = tuple(ch_mults)
        self.attn_resolutions = tuple(attn_resolutions)
        self.num_classes = num_classes
        self.num_res_blocks = num_res_blocks
        self.image_size = image_size
        self.d_emb = d_emb if d_emb is not None else 4 * base_ch

        # --- SOLUTION: TODO 3 (conditioning path) ------------------------
        # The ladder's width is base_ch; the MLP lifts it to d_emb. The label
        # table has the ladder's width so it can be SUMMED into emb(t)
        # BEFORE the projection (U0.T4 @eq-embsum) — one shared projection,
        # whatever the number of conditioning signals.
        self.emb_mlp = nn.Sequential(
            nn.Linear(base_ch, self.d_emb),
            nn.SiLU(),
            nn.Linear(self.d_emb, self.d_emb),
        )
        self.label_emb = (
            nn.Embedding(num_classes, base_ch) if num_classes is not None else None
        )
        # --- END SOLUTION (TODO 3) ---

        # --- SOLUTION: TODO 2 (wire the architecture) --------------------
        # Encoder. `skip_chs` records the channel count of every activation
        # pushed onto the skip stack, because the decoder concatenates them
        # and its input width depends on what the encoder happened to push.
        # Getting this table right IS the honest difficulty of the lab.
        self.conv_in = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)

        self.down = nn.ModuleList()
        skip_chs = [base_ch]
        ch = base_ch
        n_levels = len(self.ch_mults)
        for level, mult in enumerate(self.ch_mults):
            for _ in range(num_res_blocks):
                block = nn.ModuleList(
                    [ResBlock(ch, base_ch * mult, self.d_emb, dropout)]
                )
                ch = base_ch * mult
                if self._level_resolution_is_attn(level):
                    block.append(AttnBlock(ch))
                self.down.append(block)
                skip_chs.append(ch)
            if level < n_levels - 1:
                self.down.append(nn.ModuleList([Downsample(ch)]))
                skip_chs.append(ch)

        # Bottleneck: the lowest resolution, where attention is affordable.
        self.mid_block1 = ResBlock(ch, ch, self.d_emb, dropout)
        self.mid_attn = AttnBlock(ch)
        self.mid_block2 = ResBlock(ch, ch, self.d_emb, dropout)

        # Decoder. One extra residual block per level (the standard "+1"):
        # the level's last skip is the one the downsample pushed, and it
        # needs a block of its own to be consumed.
        self.up = nn.ModuleList()
        for level, mult in reversed(list(enumerate(self.ch_mults))):
            for i in range(num_res_blocks + 1):
                # CONCATENATION, not addition (U0.T4 @sec-unet): the decoder
                # receives the encoder's features as extra channels and
                # learns what to do with them.
                block = nn.ModuleList(
                    [
                        ResBlock(
                            ch + skip_chs.pop(), base_ch * mult, self.d_emb, dropout
                        )
                    ]
                )
                ch = base_ch * mult
                if self._level_resolution_is_attn(level):
                    block.append(AttnBlock(ch))
                if level > 0 and i == num_res_blocks:
                    block.append(Upsample(ch))
                self.up.append(block)

        self.norm_out = _norm(ch)
        # Zero-init the output convolution: the untrained field is the zero
        # field, which Part C ritual 2 asserts.
        self.conv_out = _zero_module(
            nn.Conv2d(ch, in_ch, kernel_size=3, padding=1)
        )
        # --- END SOLUTION (TODO 2) ---

    def _level_resolution_is_attn(self, level: int) -> bool:
        """Does level `level` carry attention?

        `attn_resolutions` is written in pixels, but a module list knows only
        its level index, so the two are reconciled here — at construction
        time, because whether the block exists is a structural question and
        not a per-batch one. Level `k` runs at `image_size // 2**k` pixels.
        A named resolution the ladder never reaches is silently unused,
        which is why `describe()` prints the levels it actually built.
        """
        return (self.image_size // (2**level)) in self.attn_resolutions

    def describe(self) -> str:
        """One line per level: resolution, width, and whether it has attention.

        The shape table of Part B, printed by the model instead of copied by
        hand. A resolution named in `attn_resolutions` that the ladder never
        reaches leaves no mark anywhere else, and this is where it shows up.
        """
        lines = [
            f"UNet  in_ch={self.in_ch}  base_ch={self.base_ch}  "
            f"d_emb={self.d_emb}  classes={self.num_classes}"
        ]
        for level, mult in enumerate(self.ch_mults):
            res = self.image_size // (2**level)
            attn = "attn" if self._level_resolution_is_attn(level) else "—"
            lines.append(
                f"  level {level}: {res:>3d}x{res:<3d}  "
                f"ch={self.base_ch * mult:<4d} {attn}"
            )
        n = sum(p.numel() for p in self.parameters())
        lines.append(f"  parameters: {n:,}")
        return "\n".join(lines)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """(B,C,H,W), (B,) float in [0,1], optional (B,) long -> (B,C,H,W)."""
        # The three laws of the contract, checked at the door. They are
        # asserts and not comments because the failure they prevent is
        # silent: a (B,1) time tensor broadcasts, trains, and produces a
        # plausible wrong model.
        if t.ndim != 1 or t.shape[0] != x.shape[0]:
            raise ValueError(
                f"t must be (B,) with B={x.shape[0]}, got {tuple(t.shape)}"
            )
        if not t.dtype.is_floating_point:
            raise TypeError(
                f"t must be a float tensor in [0,1] (FM arrow), got {t.dtype}"
            )
        if float(t.min()) < 0.0 or float(t.max()) > 1.0:
            raise ValueError(
                "t must lie in [0,1] — t=0 is noise, t=1 is data. Convert any "
                f"other time variable outside this network. Got "
                f"[{float(t.min()):.4g}, {float(t.max()):.4g}]."
            )

        emb = timestep_embedding(t, self.base_ch)
        if y is not None:
            if self.label_emb is None:
                raise ValueError(
                    "y was given but this UNet was built with num_classes=None"
                )
            # Summed at the ladder's width, BEFORE the projection
            # (U0.T4 @eq-embsum). One shared projection, so the modulation
            # path does not grow when a conditioning signal is added.
            emb = emb + self.label_emb(y)
        c = self.emb_mlp(emb)

        h = self.conv_in(x)
        skips = [h]
        for block in self.down:
            for layer in block:
                h = layer(h, c) if isinstance(layer, ResBlock) else layer(h)
            skips.append(h)

        h = self.mid_block2(self.mid_attn(self.mid_block1(h, c)), c)

        for block in self.up:
            h = torch.cat([h, skips.pop()], dim=1)
            for layer in block:
                h = layer(h, c) if isinstance(layer, ResBlock) else layer(h)

        return self.conv_out(F.silu(self.norm_out(h)))


# ===========================================================================
# `vae-encoder-adapter` — the PS0-only bridge (L3-U0-L4 section 2A, frozen)
# ===========================================================================


class VAEEncoderAdapter(nn.Module):
    """Present a U-Net's DOWN PATH as a VAE encoder, without touching `UNet`.

        VAEEncoderAdapter(unet_down_path, latent_dim)
        forward(x: (B,C,H,W) float) -> (mu: (B,d_z), logvar: (B,d_z))

    **Why this class exists.** PS0 asks for the `vae-mnist-scratch` encoder
    contract of U0.L3 — `encoder(x) -> (mu, logvar)` — rebuilt from the
    U-Net's own down-path blocks. The obvious shortcut, handing `UNet` itself
    to the VAE, does not typecheck against either contract: `UNet.forward`
    demands a time argument and returns a field of the input's shape, while
    a VAE encoder takes no time and returns two latent vectors. This adapter
    is the legal bridge. It changes nothing about `UNet.forward`, which stays
    the field interface U3.L2 trains.

    **What it reuses, and what it adds.** It holds the SAME module objects as
    the U-Net's down path — `conv_in`, the `down` stack, and the bottleneck —
    so the parameters are shared, not copied, and the configuration
    vocabulary (`base_ch`, `ch_mults`, `attn_resolutions`) is the U-Net's.
    It adds exactly two things: a learned constant conditioning vector, and
    a pooling/flatten projection to the two VAE heads.

    **The constant conditioning vector.** The provided `ResBlock` takes a
    conditioning vector `c` because the field network is time-conditioned.
    A VAE encoder has no time, so the adapter supplies one learned vector,
    the same for every example. That keeps the blocks unchanged — and the
    alternative, stripping FiLM out of the shared blocks, would fork the
    very parameters PS0 is supposed to share.

    **Head initialization** follows the U0.L3 rule: zero biases with small
    nonzero weights, so `mu ~ 0` and `logvar ~ 0` at the start (the latent
    begins essentially standard normal) while the first optimizer step still
    reaches the trunk.
    """

    def __init__(self, unet_down_path: "UNet", latent_dim: int) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        u = unet_down_path
        # Shared modules, not copies: PS0's point is that these are the same
        # blocks the field network uses.
        self.conv_in = u.conv_in
        self.down = u.down
        self.mid_block1, self.mid_attn, self.mid_block2 = (
            u.mid_block1, u.mid_attn, u.mid_block2
        )
        self.base_ch, self.ch_mults = u.base_ch, u.ch_mults
        self.attn_resolutions = u.attn_resolutions
        self.latent_dim = latent_dim

        # No time input: one learned conditioning vector, shared by every
        # example, standing where emb_mlp(emb(t)) stands in the field network.
        self.const_cond = nn.Parameter(torch.zeros(u.d_emb))

        bottleneck_ch = u.base_ch * u.ch_mults[-1]
        self.norm_out = _norm(bottleneck_ch)
        self.mu = nn.Linear(bottleneck_ch, latent_dim)
        self.logvar = nn.Linear(bottleneck_ch, latent_dim)
        for head in (self.mu, self.logvar):
            nn.init.normal_(head.weight, std=1e-3)   # small, NOT zero (U0.L3)
            nn.init.zeros_(head.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(B,C,H,W) float -> (mu, logvar), each (B, latent_dim)."""
        c = self.const_cond.unsqueeze(0).expand(x.shape[0], -1)
        h = self.conv_in(x)
        for block in self.down:
            for layer in block:
                h = layer(h, c) if isinstance(layer, ResBlock) else layer(h)
        h = self.mid_block2(self.mid_attn(self.mid_block1(h, c)), c)
        # Explicit global pooling, then the two heads. Pooling rather than a
        # flatten so the adapter does not depend on the input resolution.
        h = F.silu(self.norm_out(h)).mean(dim=(2, 3))
        return self.mu(h), self.logvar(h)
