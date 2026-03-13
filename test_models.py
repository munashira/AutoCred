from google import genai

client = genai.Client(api_key="AIzaSyBmzywW-raY1EiJARaEgKqusxyfsCYUXPg")

for model in client.models.list():
    print(model.name)