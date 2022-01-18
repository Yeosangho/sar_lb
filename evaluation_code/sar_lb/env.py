from flask import Flask,request,Response, jsonify
import Config as config
import util
from collections import deque
import random
import time
Traffic_Data = {}
app = Flask(__name__)

global topo_inited, changing_topo
global global_tick
import redis
import requests
changing_topo = False
topo_inited = True
global_tick = 0

SWITCHES_ROLE = {}
Switch_traffic = {}
controller_ips = config.SWITCHES.keys()

max_retry_num = 5
r = redis.Redis(host='210.107.197.219', port=6379, db=0)
r.set('foo', 'bar')
print(r.get('foo'))

@app.route("/")
def hello():
    return "This is the Monitor Node!"



@app.route(config.MONITOR['METHODS']['FINISH_MIGRATION'][0], methods=[config.MONITOR['METHODS']['FINISH_MIGRATION'][1]])
def change_topo():
    content = request.get_json(silent=True)
    global changing_topo

    # after migration, update the Topology here
    source_ctrl = content['sourceController']
    dest_ctrl = content['targetController']
    switch_id = content['targetSwitch']
    if SWITCHES_ROLE[source_ctrl][int(switch_id)] == 's':
        SWITCHES_ROLE[dest_ctrl][int(switch_id)] = 'm'
    elif SWITCHES_ROLE[source_ctrl][int(switch_id)] == 'm':
        SWITCHES_ROLE[dest_ctrl][int(switch_id)] = 's'


@app.route(config.MONITOR['METHODS']['TOPO_REPORT'][0],methods=[config.MONITOR['METHODS']['TOPO_REPORT'][1]])
def gen_topo():
    global topo_inited, Switch_traffic
    content = request.get_json(silent=True)

    Switch_traffic = {}
    #print(content)
    Traffic_Data[content['ctrl']] = [{
        'switch_id':id,
        'traffic':{} ## TODO:
        ## Design the traffic structure
    }  for id in content['switches']]

    if len(Traffic_Data) == len(config.CONTROLLERS):
        for key in Traffic_Data:
            SWITCHES_ROLE[key] = config.SWITCHES[key]
            response = util.Http_Request(key + config.CONTROLLER['METHODS']['INIT_ROLE'][0],config.SWITCHES[key])

        topo_inited = True
        print("Topology Inited")
        
    return jsonify(**{
        'success':True
    })

@app.route("/init", methods=["GET"])
def env_init():
    #init for test
    SWITCHES_ROLE = config.SWITCHES
    print(SWITCHES_ROLE['http://210.107.197.213:8080'])
    return None

@app.route("/step/<int:switch_id>/<int:controller_id>", methods=["GET"])
def step(switch_id, controller_id):
    #각 스텝마다, Switch를 하나씩 순회하며 확인
    #check current source_ctrl
    source_ctrl_id = 0
    for i in range(len(ctorller_ips)):
        if(SWITCHES_ROLE[controller_ips[i]][switch_id] == 'm'):
            source_ctrl_id = i
            break

    #current source_ctrl === contrller_id 인 경우, migration 안해도됨. 


    #아닌 경우 migration 수행. 
    if(source_ctrl_id is not contrller_id):
        obj = {
            'source' : controller_ips[source_ctrl_id],
            'dest' : controller_ips[controller_id],
            'switch' : switch_id
        }
        response_text = 'start'
        retry_num = 0
        while ('retry' in response_text or 'start' in response_text) and(retry_num < max_retry_num):

            #response = util.Http_Request(source + config.CONTROLLER['METHODS']['START_MIGRATION'][0], obj)




            #response_text = response.text
            response_text = "success"
            if('retry' in response_text):
                time.sleep(1)
                retry_num = retry_num + 1
    r.get('foo')
                
    if(retry_num == max_retry_num):
        return jsonify({"sucess" :  False}), 500
    else :
        return jsonify({"success": True, "state" : state, "reward" : reward }), 200 


def monitor():
    global topo_inited, changing_topo
    global global_tick
    # print "This is the monitor thread"
    if topo_inited and not changing_topo:
        # Implement the monitoring algorithm here, and notify controller use util.HTTP_Request
        # {controller_ip: {switch_id: number,},}


        controller_ip_1 = config.SWITCHES.keys()[0]
        controller_ip_2 = config.SWITCHES.keys()[1]
        controller_ip_3 = config.SWITCHES.keys()[2]
        # calculate the traffic sum for two controllers

        switch_id = (global_tick % 16)+1
        global_tick = global_tick + 1  
        switch_id = str(switch_id)
        #nearest_switch_id = u'1'
        source = controller_ip_3
        dest = controller_ip_1
        if(int(switch_id) >= 8):
            source = controller_ip_2 
        obj = {
            'source' : source,
            'dest' : dest,
            'switch' : switch_id
        }
        #print(obj)        
        response_text = 'start'

        if(int(switch_id) == 16):
            changing_topo = True
        #if sum1 > sum2 * 1.5:
        #    for key in controller_traffic_1:
        #        distance = abs(controller_traffic_1[key] - delta/2)
        #        #if distance < minimum:
        #        nearest_switch_id = key
        #        minimum = distance
        #    obj = {
        #        'source' : controller_ip_1,
        #        'dest' : controller_ip_2,
        #        'switch' : nearest_switch_id
        #    }
        #    print obj
        #    util.Http_Request(controller_ip_1 + config.CONTROLLER['METHODS']['START_MIGRATION'][0], obj)
        #    changing_topo = False
#
        #elif sum2 > sum1 * 1.5:
        #    for key in controller_traffic_2:
        #        distance = abs(controller_traffic_2[key] - delta/2)
        #        #if distance < minimum:
        #        nearest_switch_id = key
        #        minimum = distance
        #    obj = {
        #        'source' : controller_ip_2,
        #        'dest' : controller_ip_1,
        #        'switch' : nearest_switch_id
        #    }
        #    print obj
        #    util.Http_Request(controller_ip_2 + config.CONTROLLER['METHODS']['START_MIGRATION'][0], obj)
        #    changing_topo = False
            # for testing
            # topo_inited = False
        
    else:
        # if the topology is not inited, then do nothing
        pass


# MAIN
#gen_topo()

app.run(host='210.107.197.219', port=config.MONITOR['PORT'])
