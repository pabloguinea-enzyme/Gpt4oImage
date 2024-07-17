import os
import streamlit as st
from openai import OpenAI
import base64

# Function to calculate relevance between news text and image description using GPT-4o
def calculate_relevance(client, news_text, image_description):
    prompt = f"Given the news text: '{news_text}', rate the relevance of the following image description on a scale from 0 to 1. Only respond with a number between 0 and 1 without any additional text. Image description: '{image_description}'"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=10
    )
    
    # Extract and return the relevance score
    relevance_score = response.choices[0].message.content.strip()
    
    # Attempt to convert the relevance score to a float
    try:
        relevance_score = float(relevance_score)
    except ValueError:
        relevance_score = 0.0  # Default to 0 if parsing fails

    return relevance_score

# Function to get image description
def get_image_description(client, uploaded_file, prompt):
    # Encode the uploaded image in base64
    encoded_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

    # Create the GPT-4o API request
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    # Extract and return the description
    return response.choices[0].message.content

# Function to summarize the news text
def summarize_text(client, text):
    prompt = f"Please summarize the following news text: '{text}'"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=150
    )
    
    summary = response.choices[0].message.content.strip()
    return summary

# Function to detect the language of the text
def detect_language(client, text):
    prompt = f"Detect the language of the following text: '{text}'"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=10
    )
    
    language = response.choices[0].message.content.strip()
    return language

# Function to translate text to a specified language
def translate_text(client, text, target_language):
    prompt = f"Translate the following text to {target_language}: '{text}'"
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )
    
    translation = response.choices[0].message.content.strip()
    return translation

# Streamlit app layout
st.title("Image Relevance to News Text using GPT-4o")
st.write("Upload a news text and images to find out which images are relevant to the text.")

# Textbox for updating OpenAI API key
api_key = st.text_input("Enter your OpenAI API key", type="password")
if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY", "")

if api_key:
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)

    # Text area for news text input
    news_text = st.text_area("Enter the news text")

    # Upload multiple images button
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if news_text and uploaded_files:
        # Detect language of the news text
        language = detect_language(client, news_text)

        # Summarize the news text
        summary = summarize_text(client, news_text)
        st.write(f"Summary ({language}): {summary}")

        # Store descriptions and relevance scores
        image_relevancies = []

        for uploaded_file in uploaded_files:
            try:
                # Display the uploaded image
                st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
                st.write("Classifying...")

                # Get the image description
                description = get_image_description(client, uploaded_file, prompt="Describe the image.")
                
                # Translate description to the language of the news text
                translated_description = translate_text(client, description, language)
                st.write(f"Description ({language}): {translated_description}")

                # Calculate relevance between news text and image description
                relevance = calculate_relevance(client, news_text, description)
                image_relevancies.append({
                    'relevance': relevance,
                    'image': uploaded_file,
                    'description': translated_description
                })
            except Exception as e:
                st.error(f"Error: {e}")

        # Sort images by relevance
        image_relevancies = sorted(image_relevancies, key=lambda x: x['relevance'], reverse=True)

        st.write("Relevant Images:")
        for item in image_relevancies:
            st.image(item['image'], caption=f"Description: {item['description']} (Relevance: {item['relevance']:.2f})", use_column_width=True)
else:
    st.error("Please provide a valid OpenAI API key.")
