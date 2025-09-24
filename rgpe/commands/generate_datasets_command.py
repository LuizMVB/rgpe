from multiprocessing import Process
from . import Command
from ..services import dataset_generator_service


class GenerateDatasetsCommand(Command):
    def execute(self, *args) -> None:
        print("---Generating datasets using multiprocessing---\n")

        p_gram = Process(target=dataset_generator_service.generate_gram_points_dataset)
        p_cogram = Process(target=dataset_generator_service.generate_cogram_points_dataset)
        p_gram.start()
        p_cogram.start()
        p_gram.join()
        p_cogram.join()

        p_distances = Process(target=dataset_generator_service.generate_distances_dataset)
        p_o_shank = Process(target=dataset_generator_service.generate_o_shank_dataset)
        p_distances.start()
        p_o_shank.start()
        p_distances.join()
        p_o_shank.join()

        print("All datasets generated.\n")
