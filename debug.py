from reflectorch import *

if __name__ == '__main__':
    config_name = 'c1'
    trainer = get_trainer_by_name(config_name, load_weights=False)
    trainer.train(10)