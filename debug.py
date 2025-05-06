from reflectorch import *

if __name__ == '__main__':
    config_name = 'a_base_point_xray_conv_standard' #'a_base_point_neutron_conv_standard' 
    trainer = get_trainer_by_name(config_name, load_weights=False)
    trainer.train(1000)