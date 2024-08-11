from typing import Tuple
import xtrack as xt

class BaseWake:
    pass

    def configure_for_tracking(self, zeta_range: Tuple[float, float],
                               num_slices: int,
                               **kwargs # for multibunch compatibility
                               ) -> None:
        from xfields.beam_elements.waketracker import WakeTracker
        self._wake_tracker = WakeTracker(
            components=_expand_components(self.components),
            zeta_range=zeta_range,
            num_slices=num_slices, **kwargs)

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
        if isinstance(ww, CombinedWake):
            out_components.extend(ww.components)
        else:
            out_components.append(ww)

    return CombinedWake(components=out_components)


class CombinedWake(BaseWake):

    def __init__(self, components):
        self.components = components


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
