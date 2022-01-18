
import Config as config
import util
import requests

requests.get("http://210.107.197.219:9200/init")
switch_num = 16
switch_id = 0
controller_num = 3
switch_bw_threshold = 0
while(True):

	#부하가 일정 이상일때만 migration을 수행하고, Migration이후, 상태정보를 얻어옴. 
	if(switch_overhead > switch_bw_threshold):
		controller_id = rand.randInt(0,2)
		response = requests.get("http://210.107.197.219:9200/step/" + str(switch_id) + "/" + str(controller_id))
		state = response.state	
		reward = response.reward
		
	switch_id = switch_id + 1