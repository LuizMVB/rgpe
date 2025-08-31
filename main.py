import sys
from rgpe.services.handle_demo_service import HandleDemoService
from rgpe.services import dataset_generator_service


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
    dataset_generator_service.generate_40_features_dataset()
    # main()
