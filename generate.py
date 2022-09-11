from src.system import TextGeneratorSystem
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="model.pkl")
    parser.add_argument('--length', type=int, default=10)
    parser.add_argument('--prefix', type=str, default="")
    args = parser.parse_args()
    text_generator_system = TextGeneratorSystem()

    text = text_generator_system.generate(args.model, args.length, args.prefix)

    print(text)
