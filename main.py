import sys
from rgpe.services.handle_demo_service import HandleDemoService


def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <demo_key>")
        exit(1)

    demo_key = sys.argv[1]

    if demo_key == 'demos':
        demos = HandleDemoService.list_demos()
        print("Demos:")
        for demo in demos:
            print(f"  {demo}")
        exit(0)

    HandleDemoService.run(demo_key)
    exit(0)


if __name__ == "__main__":
    main()
