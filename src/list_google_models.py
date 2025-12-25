import google.generativeai as genai

genai.configure(api_key="AIzaSyCHmc2SbYVf2n0f_M9H0rVDwfa8lifS1hA")

models = genai.list_models()

for model in models:
    print(model.name, model.supported_generation_methods)
