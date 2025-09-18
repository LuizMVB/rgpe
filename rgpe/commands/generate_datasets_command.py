from . import Command
from ..services import dataset_generator_service


class GenerateDatasetsCommand(Command):
    def execute(self, *args) -> None:
        print("---Generating datasets---\n")
        dataset_generator_service.generate_gram_points_dataset()
        dataset_generator_service.generate_distances_dataset()
        dataset_generator_service.generate_o_shank_dataset()
        print("Done.\n")
