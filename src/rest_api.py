import json
import requests
import sys

def get_url(model_name, host='127.0.0.1',port='8501', verb='predict', version=None):
	url =f"http://{host}:{port}/v1/models/{model_name}"
	if version:
		url+="versions/{version}"
	url += f":{verb}"
	return url 


def get_model_predict(model_input, model_name="saved_model", signature_name='serving_default'):
	url = get_url(model_name)
	data = {"instances" : [model_input]}
	
	rv = requests.post(url, data=json.dumps(data))
	if rv.status_code != requests.codes.ok:
		rv.raise_for_status()
	return rv.json()["predictions"]

	
if __name__ == "__main__":
	
	while True:
		print("Enter new review:")
		if sys.version_info[0]<=3:
			sentence = input()
		if sentence  == ':q':
			break
		model_input = sentence
		model_prediction = get_model_predict(model_input)
		print(model_prediction)
		
