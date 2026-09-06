# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

import xtrack as xt
import xwakes as xw
import numpy as np
import pytest
from xobjects.test_helpers import for_all_test_contexts

class DummyCommunicator:
    def __init__(self, rank, size):
        self.rank = rank
        self.size = size

    def Get_size(self):
        return self.size

    def Get_rank(self):
        return self.rank


def test_filling_scheme_compatibility_alias():
    wake = xw.WakeResonator(kind='dipole_x', r=1e9, q=5, f_r=20e6)
    wake.configure_for_tracking(
        zeta_range=(-1, 1), num_slices=2,
        filling_scheme=[1, 0, 1], bunch_spacing_zeta=5,
        num_turns=1, circumference=20)
    assert list(wake.slicer.filled_slots) == [0, 2]

    with pytest.raises(ValueError, match='Only one'):
        wake.configure_for_tracking(
            zeta_range=(-1, 1), num_slices=2,
            filling_pattern=[1, 0, 1], filling_scheme=[1, 0, 1],
            bunch_spacing_zeta=5, num_turns=1, circumference=20)

    source_slots = np.array([2, 0], dtype=np.int64)
    wake.configure_for_tracking(
        zeta_range=(-1, 1), num_slices=2,
        filled_slots=source_slots, num_slots=4,
        bunch_spacing_zeta=5, num_turns=1, circumference=20)
    source_slots[:] = 1
    exposed_slots = wake.filled_slots
    exposed_slots[:] = 1
    assert list(wake.filled_slots) == [0, 2]
    assert list(wake.filling_pattern) == [1, 0, 1, 0]


@for_all_test_contexts
def test_config_pipeline_for_wakes(test_context):
    comm_size = 3
    for my_rank in range(comm_size):
        communicator = DummyCommunicator(rank=my_rank, size=comm_size)

        zeta_range = (-1, 1)
        num_slices = 10
        filling_pattern = [1, 0, 1, 1, 1]
        bunch_spacing_zeta = 5
        num_turns = 1
        circumference = 100

        wake1 = xw.WakeResonator(kind='dipole_x', r=1e9, q=5, f_r=20e6)

        wake1.configure_for_tracking(
            zeta_range=zeta_range,
            num_slices=num_slices,
            filling_pattern=filling_pattern,
            bunch_spacing_zeta=bunch_spacing_zeta,
            num_turns=num_turns,
            circumference=circumference,
            _context=test_context
        )

        wake2 = xw.WakeResonator(kind='dipole_x', r=1e9, q=5, f_r=20e6)

        wake2.configure_for_tracking(
            zeta_range=zeta_range,
            num_slices=num_slices,
            filling_pattern=filling_pattern,
            bunch_spacing_zeta=bunch_spacing_zeta,
            num_turns=num_turns,
            circumference=circumference,
            _context=test_context
        )

        wake3 = wake1 + wake2

        wake3.configure_for_tracking(
            zeta_range=zeta_range,
            num_slices=num_slices,
            filling_pattern=filling_pattern,
            bunch_spacing_zeta=bunch_spacing_zeta,
            num_turns=num_turns,
            circumference=circumference,
            _context=test_context
        )

        wake_no_config = xw.WakeResonator(kind='dipole_x', r=1e9, q=5, f_r=20e6)

        drift = xt.Drift(length=1)

        line = xt.Line(elements=[wake1, wake2, wake3, wake_no_config, drift])

        particles = xt.Particles(_context=test_context)

        pipeline_manager = xw.config_pipeline_for_wakes(particles, line, communicator,
                                    elements_to_configure=None)

        assert len(pipeline_manager._IDs) == comm_size
        assert len(pipeline_manager._particles_per_rank) == comm_size

        for ii in range(comm_size):
            assert list(pipeline_manager._IDs.keys())[ii] == f'particles{ii}'
            assert pipeline_manager._particles_per_rank[ii] == [f'particles{ii}']

        assert len(pipeline_manager._elements) == 3
        assert pipeline_manager._elements['e0'] == 0
        assert pipeline_manager._elements['e1'] == 1
        assert pipeline_manager._elements['e2'] == 2

        assert pipeline_manager._communicator == communicator
        assert pipeline_manager.verbose == False
        assert pipeline_manager._pending_requests == {}
        assert pipeline_manager._last_request_turn == {}

        assert wake1._wake_tracker.partner_names == [f'particles{rank}'
                                for rank in range(comm_size) if rank != my_rank]
        assert wake2._wake_tracker.partner_names == [f'particles{rank}'
                                for rank in range(comm_size) if rank != my_rank]
        assert wake3._wake_tracker.partner_names == [f'particles{rank}'
                                for rank in range(comm_size) if rank != my_rank]

        assert particles.name == f'particles{my_rank}'
