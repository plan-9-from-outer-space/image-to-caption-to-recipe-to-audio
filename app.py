import os
from dotenv import load_dotenv
import openai
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
# allows us to download a huggingface model into our local machine
from transformers import pipeline
import requests
import streamlit as st

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
HUGGINGFACE_HUB_API_TOKEN = os.getenv("HUGGINGFACE_HUB_API_TOKEN")

# 1. Image to text implementation (aka image captioning) with huggingface
def image_to_text(url):
    pipe = pipeline(
        task = "image-to-text",
        model = "Salesforce/blip-image-captioning-large",
        max_new_tokens = 1000)
    text = pipe(url)[0]["generated_text"]
    print(f"Image Captioning:: {text}")
    return text


# 2. llm - generate a recipe from the image text
llm_model = "gpt-3.5-turbo"  # or gpt4, but gpt3.5 is cheaper!
llm = ChatOpenAI(temperature=0.7, model=llm_model)


def generate_recipe(ingredients):
    template = """\
                You are a extremely knowledgeable nutritionist, bodybuilder and chef who also knows
                everything one needs to know about the best quick, healthy recipes. 
                You know all there is to know about healthy foods, healthy recipes that keep 
                people lean and help them build muscles, and lose stubborn fat.
                
                You've also trained many top performers athletes in body building, and in extremely 
                amazing physique. 
                
                You understand how to help people who don't have much time and or 
                ingredients to make meals fast depending on what they can find in the kitchen. 
                Your job is to assist users with questions related to finding the best recipes and 
                cooking instructions depending on the following variables:
                0/ {ingredients}
                
                When finding the best recipes and instructions to cook,
                you'll answer with confidence and to the point.
                Keep in mind the time constraint of 5-10 minutes when coming up
                with recipes and instructions as well as the recipe.
                
                If the {ingredients} are less than 3, feel free to add a few more
                as long as they will compliment the healthy meal.
                
                Make sure to format your answer as follows:
                - The name of the meal as bold title (new line)
                - Best for recipe category (bold)
                    
                - Preparation Time (header)
                    
                - Difficulty (bold):
                    Easy
                - Ingredients (bold)
                    List all ingredients 
                - Kitchen tools needed (bold)
                    List kitchen tools needed
                - Instructions (bold)
                    List all instructions to put the meal together
                - Macros (bold): 
                    Total calories
                    List each ingredient calories
                    List all macros 
                    
                Please make sure to be brief and to the point.  
                Make the instructions easy to follow and step-by-step.
    """
    prompt = PromptTemplate(template=template, input_variables=["ingredients"])
    recipe_chain = LLMChain(llm=llm, prompt=prompt, verbose=True)
    recipe = recipe_chain.run(ingredients)
    # Identical call methods:
    # recipe = recipe_chain({"ingredients": ingredients})
    # recipe = recipe_chain.__call__({"ingredients": ingredients})
    # recipe = recipe_chain.predict(ingredients=ingredients)

    return recipe


# 3. Text to speech
def text_to_speech(text):
    # From https://huggingface.co/facebook/fastspeech2-en-ljspeech
    #   then click on Deploy -> Inference API
    API_URL = "https://api-inference.huggingface.co/models/facebook/fastspeech2-en-ljspeech"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_HUB_API_TOKEN}"}
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.content


def main():

    # # Step 1
    # caption = image_to_text(url='mango_fruits.jpeg')
    # print(caption)

    # # Step 2
    # audio = text_to_speech(text="This is a sentence.")
    # audio = text_to_speech(text=caption)
    # with open("audio.flac", "wb") as file:
    #     file.write(audio)

    # # Step 3
    # # caption = 'a close up of a table with bowls of fruit and a bowl of milk'
    # recipe = generate_recipe(ingredients=caption)
    # print('recipe = ', recipe)

    # exit()

    st.title(":robot_face: Image To Recipe 👨🏾‍🍳")
    st.header("Upload a food/meal image and get a recipe")

    upload_file = st.file_uploader(
        label = "Choose a food or meal image:", 
        type = ["jpg", "png"])

    if upload_file is not None:
        print(upload_file)
        file_bytes = upload_file.getvalue()
        with open(upload_file.name, "wb") as file:
            file.write(file_bytes)

        st.image(
            image = upload_file,
            caption = "The uploaded image file",
            use_column_width = True,
            width = 250)

        ingredients = image_to_text(upload_file.name)

        audio = text_to_speech(ingredients)
        with open("audio.flac", "wb") as file:
            file.write(audio)

        recipe = generate_recipe(ingredients=ingredients)

        with st.expander("Ingredients"):
            st.write(ingredients)
        with st.expander("Recipe"):
            st.write(recipe)

        st.audio("audio.flac")


# Invoking main function
if __name__ == "__main__":
    main()


#####################################################
### EXAMPLE RECIPE OUTPUT
#####################################################

# recipe =  - **Fruit and Milk Bowl**
#
# - Best for recipe category: Breakfast or Snack
#
# - Preparation Time:
#     5 minutes
#
# - Difficulty:
#     Easy
#
# - Ingredients:
#     - Assorted fruits (e.g., berries, banana, apple, orange)
#     - Milk (dairy or plant-based)
#
# - Kitchen tools needed:
#     - Bowl
#     - Spoon
#
# - Instructions:
#     1. Wash and chop the fruits into bite-sized pieces.
#     2. Place the fruits in a bowl.
#     3. Pour milk over the fruits until desired amount is achieved.
#     4. Mix gently with a spoon to combine the fruits and milk.
#     5. Enjoy immediately.
#
# - Macros:
#     - Total calories: Depends on the amount and type of fruits and milk used.
#     - Calories from fruits: Varies based on the specific fruits used.
#     - Calories from milk: Varies based on the type and quantity of milk used.
#     - Macros will vary depending on the specific fruits and milk used.
