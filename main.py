from trainer import Trainer
import config


def main():
    if config.train:
        trainer = Trainer()
        trainer.train()


if __name__ == "__main__":
    main()