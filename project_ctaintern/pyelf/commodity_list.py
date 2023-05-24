
comms_agri = ['a', 'ap', 'b', 'c', 'cf', 'cy', 'cs', 'jd', 'lh', 'm', 'oi', 'p', 'pk', 'rm', 'sr', 'yy', 'cj']#17
comms_black = ['hc', 'i', 'j', 'jm', 'rb', 'sf', 'sm', 'ss']#9, 'zc'
comms_chem = ['bu', 'eb', 'eg', 'fg', 'fu', 'l', 'lu', 'ma', 'nr', 'pf', 'pg', 'pp', 'ru', 'sa', 'sc', 'sp', 'ta', 'ur', 'v']#19
comms_metal = ['ag', 'al', 'au', 'cu', 'ni', 'pb', 'sn', 'zn','si']#8

comms_all = comms_agri+comms_black+comms_chem+comms_metal

Sector_dict_1sec = {}
Sector_dict_1sec['all'] = comms_all

Sector_dict_2sec = {}
Sector_dict_2sec['industry'] = comms_black+comms_chem+comms_metal
Sector_dict_2sec['agri'] = comms_agri

Sector_dict_3sec = {}
Sector_dict_3sec['blackmetal'] = comms_black+comms_metal
Sector_dict_3sec['chemical'] = comms_chem
Sector_dict_3sec['agri'] = comms_agri

Sector_dict_4sec = {}
Sector_dict_4sec['black'] = comms_black
Sector_dict_4sec['metal'] = comms_metal
Sector_dict_4sec['chemical'] = comms_chem
Sector_dict_4sec['agri'] = comms_agri