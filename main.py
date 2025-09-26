import sys
from rgpe import cli


def main():
    if len(sys.argv) < 2:
        cli.display_help()
        exit(0)

    command_key = sys.argv[1]
    args = sys.argv[2:]
    cli.run(command_key, *args)


if __name__ == "__main__":
    main()
