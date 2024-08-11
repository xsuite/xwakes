from xtrack.pipeline.manager import PipelineManager
from xtrack.pipeline.multitracker import PipelineBranch, PipelineMultiTracker

def config_pipeline_manager_and_multitracker_for_wakes(particles, line,
                                                       wakes_dict,
                                                       communicator):

    pipeline_manager=PipelineManager(communicator)

    comm_size = communicator.Get_size()
    my_rank = communicator.Get_rank()

    for rank in range(comm_size):
        pipeline_manager.add_particles(f'particles{rank}', rank)

    particles.init_pipeline(f'particles{my_rank}')

    for wf_name, wf in wakes_dict.items():
        pipeline_manager.add_element(wf_name)
        if hasattr(wf, '_wake_tracker'):
            #for wake elements
            wf._wake_tracker.init_pipeline(
                pipeline_manager=pipeline_manager,
                element_name=wf_name,
                partners_names=[f'particles{rank}'
                                for rank in range(comm_size) if rank != my_rank])
        else:
            #for other elements with slicer, e.g. collective monitor
            wf.init_pipeline(
                pipeline_manager=pipeline_manager,
                element_name=wf_name,
                partners_names=[f'particles{rank}'
                                for rank in range(comm_size) if rank != my_rank])

    return pipeline_manager
