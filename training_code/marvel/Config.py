"""
Configuration
"""
NODE_NUMBER_PER_SWITCH =1     # N1-N5
CONTROLLERS = [
    {
        'name':"C1",
        'ip':'210.107.197.213',
        'port':6633
    },
    {
        'name':"C2",
        'ip':'210.107.197.213',
        'port':6634
    },    
    {
        'name':"C3",
        'ip':'210.107.197.211',
        'port':6633
    }
]                 # sn1 + sn2 = 8
#SWITCHES = {
#    'http://210.107.197.213:8080':['m','m','m','m','m','m','m','m'],
#    'http://210.107.197.211:8080':['s','s','s','s','s','s','s','s']
#}
SWITCHES = {
    'http://210.107.197.213:8080':['m','m','m','m','m','m','m','s','s','s','s','s','s','s','s','s'],
    'http://210.107.197.213:8081':['s','s','s','s','s','s','s','m','m','m','m','m','m','m','m','m'],
    'http://210.107.197.211:8080':['s','s','s','s','s','s','s','s','s','s','s','s','s','s','s','s']
}
SWITCH_NUMBER = 16



MONITOR = {
    "PORT":9200,
    "METHODS":{
        "STAT":["/stat","POST"],
        "FINISH_MIGRATION":["/change_topo","POST"],
        "TOPO_REPORT":["/report_topo","POST"]
    },
    "CHECK_INTERVAL":4         # seconds
}

CONTROLLER = {
    'STAT_SUBMIT_INTERVAL':2,
    'METHODS':{
        'INIT_ROLE':['/init_role','POST'], # because the util.Http_Request only use POST
        'START_MIGRATION':['/migrate','POST'],
    'MIGRATION_BEGIN':['/begin','POST'],
    'MIGRATION_READY':['/ready','POST'],
    'MIGRATION_END':['/end','POST']
    }
}



USE_TOPO = 'ye'
#USE_TOPO = 'tree'

TOPO = {}
TOPO['tree'] = [[1,2],[1,3],
                [2,4],[2,5],
                [3,6],[3,7],[3,8]];

TOPO['cube'] = [[1,2],[1,4],[1,5],
                [2,3],[2,6],
                [3,7]];

TOPO['ye'] = [[2,1],
              [3,2],[3,4],[3,5],
              [4,6],
              [6,7],
              [7,8],
              [7,10],
              [8,9],
              [9,11],
              [11,12],[11,14],
              [12,13],
              [14,15],
              [15,16]
              ];

