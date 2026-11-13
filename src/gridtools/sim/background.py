"""Real backgrounds for hybrid scenes (the scene-facing API).

Hybrid scenes add stretches of real grid recordings with few or no chirps
to simulated fish, at a controlled SNR, so that the detector and the
assignment model see real noise (hum harmonics, clicks, non-stationary
interference) and real, tracked, non-chirping fish. The library is built by
:mod:`gridtools.background` (``python -m gridtools.background build``) and
documented in its ``report.md``.

How a hybrid scene uses it
--------------------------
1. Draw the background first; it fixes the grid::

       lib = BackgroundLibrary.load("LIBRARY")
       segment = lib.sample(rng, duration_s, n_electrodes=n_electrodes,
                            samplerate_hz=20_000.0, resample=True)

   The excerpt holds ``n_electrodes`` channels: the whole grid of a
   segment with exactly that many, or a contiguous run in channel order of
   a larger one (``channels="random"`` for random subsets; dead and noisy
   channels are avoided unless ``avoid_bad_channels=False``), so a
   single-electrode scene can use any segment. The scene then uses
   ``segment.geometry`` (the chosen electrodes' positions),
   ``segment.samplerate_hz`` and ``segment.n_samples``, and never assigns
   the channels to positions of its own: v1 put random electrode subsets
   on a fixed simulated grid, which scrambled the geometry (audit G7).
   Positions are nominal for the competition grids
   (``segment.positions_nominal``: their ``fishgrid.cfg`` spacing is
   disputed, the true one unknown) and exact for the field grids with a
   layout file. ``segment.channels`` lists the chosen source channels,
   ``segment.tracks`` carries ``sign_v`` columns for exactly them. With
   ``avoid_bad_channels=False``, give simulated fish zero gain on
   ``segment.dead_channels`` (no fish signal there in the source).
2. Simulate the fish on that grid and compute their signal level from the
   noise-free fundamental amplitudes on every electrode (amplitude x
   relative amplitude of the fundamental x field gain x channel gain), e.g.
   on the scene's track frames::

       fundamentals = [model.gain_track(i).evaluate(t) * model.channel_gain
                       * fish.amplitude * fish.harmonics.amplitudes[0]
                       for i, fish in enumerate(model.fish)]
       signal = fish_signal_level(fundamentals)

3. Draw a target SNR and scale the background::

       real = segment.info["real_snr_max_db"]  # NaN without real fish
       snr = SnrDistribution().sample(
           rng, min_db=None if math.isnan(real) else real - 30.0)
       scale = scale_for_snr(segment.level, signal, snr)
       recording = add_background(recording, segment, scale,
                                  geometry=segment.geometry, samplerate_hz=fs)

   Real fish keep their own SNR when the background is scaled, so a hard
   target leaves them far louder than the simulated fish; ``min_db`` caps
   that gap (30 dB covers 95 % of the level differences between two real
   fish in one segment). Add the background after the scene's channel
   gains and instead of (or in addition to, then pass
   ``other_noise_level``) the synthetic noise. ``scale`` also converts the
   background's unit (ADC volts, or int16 full scale for field loggers)
   into the scene's.
4. Merge the real fish into the scene's tracks under identities from 1000
   on and list them in ``scene.json`` with ``real_fish=True``::

       nfft_ref = config.tracks.power_nfft
       offset = 20 * log10(scale) + 10 * log10(
           nfft_ref / segment.info["track_power_nfft"])
       tracks, real_ids = merge_tracks(scene_tracks, segment.tracks,
                                       power_offset_db=offset)
       real = segment.real_fish(real_ids)  # ident, source_ident, real_fish, ...

   Real fish never chirp in the ground truth (``chirps.csv`` lists only
   simulated chirps). Segments with ``screened=False`` (2022, no chirp
   detections) may contain real chirps; their default sampling weight is
   lower, and they should be re-screened with the new detector.
   ``tracks_complete=False`` flags segments with a line no track explains
   (possibly an untracked fish) or a fish tracked in < 90 % of the frames.
5. Record ``segment.segment_id``, ``segment.offset_s``, ``scale``, the target
   SNR and ``segment.info["split_group"]`` in ``scene.json``; split
   train/validation/test by ``split_group`` (the source recording).

Level and SNR definitions: :mod:`gridtools.background.levels`.
"""

from gridtools.background.levels import (
    LEVEL_BAND_HZ,
    REFERENCE_BANDWIDTH_HZ,
    BackgroundLevel,
    SnrDistribution,
    background_level,
    fish_signal_level,
    fish_signal_powers,
    scale_for_snr,
    snr_db,
)
from gridtools.background.library import (
    REAL_FISH_IDENT_OFFSET,
    BackgroundLibrary,
    BackgroundSegment,
    add_background,
    merge_tracks,
)

__all__ = [
    "LEVEL_BAND_HZ",
    "REAL_FISH_IDENT_OFFSET",
    "REFERENCE_BANDWIDTH_HZ",
    "BackgroundLevel",
    "BackgroundLibrary",
    "BackgroundSegment",
    "SnrDistribution",
    "add_background",
    "background_level",
    "fish_signal_level",
    "fish_signal_powers",
    "merge_tracks",
    "scale_for_snr",
    "snr_db",
]
