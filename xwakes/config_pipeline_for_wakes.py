from xtrack.pipeline.manager import PipelineManager
from xtrack.pipeline.multitracker import PipelineBranch, PipelineMultiTracker

import xfields as xf

def config_pipeline_manager_and_multitracker_for_wakes(particles, line,
                                                       communicator,
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
        elements_to_configure = []
        for nn in line.element_names:
            if hasattr(line[nn], '_wake_tracker'):
                elements_to_configure.append(nn)
            elif isinstance(line[nn], ElementWithSlicer):
                elements_to_configure.append(nn)

    for nn in elements_to_configure:
        ee = line[nn]
        pipeline_manager.add_element(nn)
        if hasattr(ee, '_wake_tracker'):
            ee = ee._wake_tracker

        ee.init_pipeline(
            pipeline_manager=pipeline_manager,
            element_name=nn,
            partners_names=[f'particles{rank}'
                            for rank in range(comm_size) if rank != my_rank])

    return pipeline_manager
