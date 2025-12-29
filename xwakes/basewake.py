# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

from typing import Tuple
import xpart as xp
import numpy as np

class Wake:

    def __init__(self, components):

        """
        Generic wake object handling multiple wake components.

        Parameters
        ----------
        components : list[xwakes.wit.Component]
            List of wake components.
        """

        self.components = components

    def configure_for_tracking(self, zeta_range: Tuple[float, float],
                               num_slices: int,
                               num_turns=1,
                               bunch_spacing_zeta=None,  # This is P in the paper
                               filling_scheme=None,
                               circumference=None,
                               bunch_selection=None,
                               **kwargs # for multibunch compatibility
                               ) -> None:

        # We import here so that xfields is not a dependency of xwakes (which
        # can be used in standalone mode for defining impedance models)
        from xfields.beam_elements.waketracker import WakeTracker
        self._wake_tracker = WakeTracker(
            components=_expand_components(self.components),
            zeta_range=zeta_range,
            num_slices=num_slices,
            num_turns=num_turns,
            bunch_spacing_zeta=bunch_spacing_zeta,
            filling_scheme=filling_scheme,
            circumference=circumference,
            bunch_selection=bunch_selection,
            **kwargs)

    def _reconfigure_for_parallel(self, n_procs, my_rank) -> None:

        filled_slots = self._wake_tracker.slicer.filled_slots
        scheme = np.zeros(np.max(filled_slots) + 1,
                        dtype=np.int64)
        scheme[filled_slots] = 1

        split_scheme = xp.matched_gaussian.split_scheme
        bunch_selection_rank = split_scheme(filling_scheme=scheme,
                                             n_chunk=int(n_procs))

        self.configure_for_tracking(zeta_range=self._wake_tracker.zeta_range,
                          num_slices=self._wake_tracker.num_slices,
                          bunch_spacing_zeta=self._wake_tracker.bunch_spacing_zeta,
                          filling_scheme=scheme,
                          bunch_selection=bunch_selection_rank[my_rank],
                          num_turns=self._wake_tracker.num_turns,
                          circumference=self._wake_tracker.circumference,
                          )

    def track(self, particles) -> None:
        if not hasattr(self, '_wake_tracker') or self._wake_tracker is None:
            raise ValueError('Wake not configured for tracking, '
                             'call `configure_for_tracking` first')
        return self._wake_tracker.track(particles)


    def __add__(self, other):
        return _add_wakes(self, other)

    def __radd__(self, other):
        return _add_wakes(other, self)

    @property
    def slicer(self):
        if hasattr(self, '_wake_tracker') and self._wake_tracker is not None:
            return self._wake_tracker.slicer
        return None

def _add_wakes(wake1, wake2):
    out_components = []
    for ww in [wake1, wake2]:
        if ww is None or ww == 0:
            continue
        if isinstance(ww, Wake):
            out_components.extend(ww.components)
        else:
            out_components.append(ww)

    return Wake(components=out_components)


def _handle_kind(kind):
    if isinstance(kind, str):
        kind = [kind]

    if isinstance(kind, (list, tuple)):
        kind = {kk: 1.0 for kk in kind}

    assert hasattr(kind, 'keys')

    return kind


def _expand_components(components):
    _expanded_components = []
    for cc in components:
        if hasattr(cc, 'function_vs_zeta'):
            _expanded_components.append(cc)
        else:
            assert hasattr(cc, 'components')
            _expanded_components.extend(
                _expand_components(cc.components))

    return _expanded_components
