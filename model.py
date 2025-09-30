from google import genai

client = genai.Client(api_key="AIzaSyA0hJO0tIhTOrEnpa1suqkOHnxy2CdVakU")

for model in client.models.list():
    print(model.name)
    print(model)              
    print("-----")
