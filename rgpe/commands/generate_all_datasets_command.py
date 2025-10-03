from multiprocessing import Process
from . import Command
from ..services import base_dataset_generator_service, custom_dataset_generator_service, j_kampe_dataset_generator_service, o_shank_dataset_generator_service


class GenerateAllDatasetsCommand(Command):
    def execute(self, *args) -> None:
        print("---Generating datasets using multiprocessing---\n")

        p_gram = Process(target=base_dataset_generator_service.generate_gram_points_dataset)
        p_cogram = Process(target=base_dataset_generator_service.generate_cogram_points_dataset)
        p_gram.start()
        p_cogram.start()
        p_gram.join()
        p_cogram.join()
        
        p_distances = Process(target=base_dataset_generator_service.generate_distances_dataset)
        p_o_shank = Process(target=o_shank_dataset_generator_service.generate_o_shank_dataset)
        p_distances.start()
        p_o_shank.start()
        p_distances.join()
        p_o_shank.join()

        p_j_kampe = Process(target=j_kampe_dataset_generator_service.generate_j_kampe_dataset, args=(100,))
        p_custom  = Process(target=custom_dataset_generator_service.generate_custom_dataset)
        p_j_kampe.start()
        p_custom.start()
        p_j_kampe.join()
        p_custom.join()

        print("All datasets generated.\n")
