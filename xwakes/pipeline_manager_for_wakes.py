from xtrack.pipeline.manager import PipelineManager
from xtrack.pipeline.multitracker import PipelineBranch, PipelineMultiTracker

class PipelineManagerForWakes(PipelineManager):
    def __init__(self, particles,
                 line,
                 wakes_dict,
                 communicator=None):

        super().__init__(communicator=communicator)

        comm_size = communicator.Get_size()
        my_rank = communicator.Get_rank()

        for rank in range(comm_size):
            self.add_particles(f'particles{rank}', rank)

        particles.init_pipeline(f'particles{my_rank}')

        for wf_name, wf in wakes_dict.items():
            self.add_element(wf_name)
            wf._wake_tracker.init_pipeline(
                pipeline_manager=self,
                element_name=wf_name,
                partners_names=[f'particles{rank}'
                                for rank in range(comm_size) if rank != my_rank])

        branch = PipelineBranch(line, particles)
        self.multitracker = PipelineMultiTracker(branches=[branch])
