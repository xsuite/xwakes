from typing import Tuple

class BaseWake:
    pass

    def configure_for_tracking(self, zeta_range: Tuple[float, float],
                               num_slices: int,
                               **kwargs # for multibuunch compatibility
                               ) -> None:
        import xfields as xf
        self._xfields_wf = xf.Wakefield(components=self.components,
                                        zeta_range=zeta_range,
                                        num_slices=num_slices, **kwargs)

    def track(self, particles) -> None:
        self._xfields_wf.track(particles)
