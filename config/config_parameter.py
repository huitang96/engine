from configparser import ConfigParser
import os

class config_file_paeser(ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr

class readconfig:
    def __init__(self):
        self.cf = config_file_paeser()
        self.config_file_path = r''
    def read_config_file(self, file_path):
        root_dir = os.path.join(file_path, "config")
        self.config_file_path = os.path.join(root_dir, "config.ini")
        try:
            self.cf.read(self.config_file_path, encoding='utf-8')
        except:
            return None
    def get_main_proc_config(self, sec):
        cfg_dict = {}
        try:
            config_list = self.cf.items(sec)
            for config_pair in config_list:
                cfg_dict[config_pair[0]] = config_pair[1]
        except:
            return None
        return cfg_dict

def config2cfg():
    configfile = readconfig()
    configfile.read_config_file("/home/thui/projects/classification_proj")
    configfile_dict = configfile.get_main_proc_config("mode1")
    cfgdict = {}
    ''' from config.ini write in '''
    cfgdict['workspace'] = configfile_dict['workspace']
    cfgdict['dataroot'] = configfile_dict['dataroot']
    cfgdict['datasave'] = configfile_dict['datasave']
    ''' from config file write in '''
    cfgdict['batch_size'] = 32
    cfgdict['learning_rate'] = 1e-3
    cfgdict['epoch'] = 50
    return cfgdict

if __name__=='__main__':
    read_ini = readconfig()
    read_ini.read_config_file(r'/home/thui/projects/classification_proj')
    test = read_ini.get_main_proc_config('mode1')
    print(test)
