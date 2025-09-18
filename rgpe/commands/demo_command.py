from . import Command
from ..services.handle_demo_service import HandleDemoService


class DemoCommand(Command):
    OPTIONS = ['list', *HandleDemoService.list_demos()]

    @classmethod
    def options(cls):
        o = ' '.join(cls.OPTIONS)
        return f"[{o}]"

    def execute(self, *args) -> None:
        option = args[0]

        if option == 'list':
            demos = HandleDemoService.list_demos()
            print("Demos:")
            for demo in demos:
                print(f"  {demo}")
            exit(0)

        HandleDemoService.run(option)
