from . import Command
from ..services.handle_demo_service import HandleDemoService


class DemoCommand(Command):
    def execute(self, option: str) -> None:
        if option == 'list':
            demos = HandleDemoService.list_demos()
            print("Demos:")
            for demo in demos:
                print(f"  {demo}")
            exit(0)

        HandleDemoService.run(option)
