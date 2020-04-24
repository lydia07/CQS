from trainer import Trainer
from inference import BeamSearcher
import config


def main():
    if config.train:
        trainer = Trainer()
        trainer.train()
    else:
        print('beam search')
        beamsearcher = BeamSearcher(config.model_path, config.output_dir)
        beamsearcher.decode()


if __name__ == "__main__":
    main()