from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="2mJz8J9Jx7NWoco65LK3"
)

result = CLIENT.infer("test.jpg", model_id="builderformer-4/2")
print(result)
