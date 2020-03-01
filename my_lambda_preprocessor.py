import json
import boto3
import datetime
import time
from Preprocessing.Preprocessing_library import Preprocess

Preprocessor=Preprocess(max_length_tweet = 20, max_length_dictionary = 1000000)
sage_maker_client = boto3.client("runtime.sagemaker")
s3 = boto3.client("s3")

def lambda_handler(event, context):
    
    tweet = event["tweet"]
    
    starttime_pre = time.time()
    features = Preprocessor.pre_process_text(tweet)
    pre_processing_time = time.time() - starttime_pre
    
    model_payload = {
        "features_input" : features
    }


    starttime_inf = time.time()
    model_response = sage_maker_client.invoke_endpoint(
        EndpointName = "sentiment-model",
        ContentType = "application/json",
        Body = json.dumps(model_payload))
    model_inference_time = time.time() - starttime_inf


    result = json.loads(model_response["Body"].read().decode())

    #def myconverter(o):
     #   if isinstance(o, datetime.datetime):
      #      return o.__str__()

    response = {}
    returned_result = {}
   
    response["date_and_time_of_the_request"] = datetime.datetime.now()
    response["tweet"] = tweet

    if result["predictions"][0][0] >= 0.5:
        response["sentiment"] = "positive"
        returned_result["sentiment"] = "positive"
    else:
        response["sentiment"] = "negative"
        returned_result["sentiment"] = "negative"

    response["probability_from_the_model"] = result["predictions"][0][0]
    response["pre_processing_time"] = pre_processing_time
    response["model_inference_time"] = model_inference_time

    filename = "model_tar/" + "payload_logging" + str(datetime.datetime.now()) + ".json"
    s3.put_object(Bucket = "twitter-cloud", Key = filename, Body = json.dumps(response, default=str, indent = 2).encode())

    print("Result: " + json.dumps(response, default=str, indent = 2))

    return returned_result