import configparser
def get_config(config_file='config.ini'):
    parser=configparser.ConfigParser()
    parser.read(config_file)
    _conf_ints = [(key,int(value)) if ',' not in value else (key,[int(i) for i in value.split(',')]) for key,value in parser.items('ints')]
    _conf_floats = [(key,float(value)) if ',' not in value else (key,[float(i) for i in value.split(',')]) for key,value in parser.items('floats')]
    _conf_strings = [(key,str(value)) for key,value in parser.items('strings')]
    return dict(_conf_ints+_conf_floats+_conf_strings)
