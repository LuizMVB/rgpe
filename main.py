import sys
from rgpe import cli


def main():
    if len(sys.argv) < 2:
        print("Error")
        exit(1)

    command_key = sys.argv[1]
    args = sys.argv[2:]
    cli.run(command_key, *args)


if __name__ == "__main__":
    main()
