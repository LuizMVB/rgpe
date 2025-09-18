from rgpe.commands import Command
from rgpe.commands.demo_command import DemoCommand
from rgpe.commands.generate_datasets_command import GenerateDatasetsCommand


commands: dict[str, type[Command]] = {
    'demo': DemoCommand,
    'generate-datasets': GenerateDatasetsCommand,
}


def display_help():
    print("Usage: python main.py [command] [?options]")
    print("Available commands:")
    for command_key, command_cls in commands.items():
        print(f'\t{command_key} {command_cls.options()}')


def run(command_key: str, *args):
    if command_key == 'help':
        display_help()
        exit(0)

    if command_key not in commands:
        raise RuntimeError(f"Unknown command: {command_key}")

    command = commands[command_key]()
    command.execute(*args)
