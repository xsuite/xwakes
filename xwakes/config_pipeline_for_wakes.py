# copyright ############################### #
# This file is part of the Xwakes Package.  #
# Copyright (c) CERN, 2024.                 #
# ######################################### #

from xtrack.pipeline.manager import PipelineManager
from xtrack.pipeline.multitracker import PipelineBranch, PipelineMultiTracker


def config_pipeline_for_wakes(particles, line, communicator,
                              elements_to_configure=None):

    assert communicator is not None, 'communicator must be provided'

    pipeline_manager=PipelineManager(communicator)

    comm_size = communicator.Get_size()
    my_rank = communicator.Get_rank()

    for rank in range(comm_size):
        pipeline_manager.add_particles(f'particles{rank}', rank)

    particles.init_pipeline(f'particles{my_rank}')

    if elements_to_configure is None:
        from xfields.beam_elements.element_with_slicer import ElementWithSlicer
        from xwakes.beam_elements.transverse_damper import TransverseDamper
        from xwakes.beam_elements.collective_monitor import CollectiveMonitor

        elements_to_configure = []
        for nn in line.element_names:
            if hasattr(line[nn], '_wake_tracker'):
                elements_to_configure.append(nn)
            elif isinstance(line[nn], ElementWithSlicer):
                elements_to_configure.append(nn)
            elif isinstance(line[nn], TransverseDamper):
                elements_to_configure.append(nn)
            elif isinstance(line[nn], CollectiveMonitor):
                elements_to_configure.append(nn)

    for nn in elements_to_configure:
        ee = line[nn]
        pipeline_manager.add_element(nn)
        if (isinstance(line[nn], TransverseDamper) or
            isinstance(line[nn], CollectiveMonitor)):
            line[nn]._reconfigure_for_parallel(comm_size, my_rank)
            continue

        if hasattr(ee, '_wake_tracker'):
            ee._reconfigure_for_parallel(comm_size, my_rank)
            ee = ee._wake_tracker

        ee.init_pipeline(
            pipeline_manager=pipeline_manager,
            element_name=nn,
            partner_names=[f'particles{rank}'
                            for rank in range(comm_size) if rank != my_rank])

    return pipeline_manager
