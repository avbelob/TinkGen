from src.system import TextGeneratorSystem
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, default=None)
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()
    text_generator_system = TextGeneratorSystem()
    text_generator_system.fit(args.input_dir, args.model)
