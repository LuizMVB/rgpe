from ..tasks import Task
from ..tasks.generate_dataset_task import GenerateDatasetTask

class HandleTaskService:
    tasks: dict[str, type[Task]] = {
        "generate_dataset": GenerateDatasetTask,
    }

    @classmethod
    def run(cls, task_key: str, *args, **kwargs):
        if task_key not in cls.tasks:
            raise KeyError(f"Task key {task_key} not found.")
        task = cls.tasks[task_key]()
        task.handle()
