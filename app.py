import os
import streamlit as st
from openai import OpenAI
import base64
from utils import get_image_description
from difflib import SequenceMatcher

# Function to calculate similarity between two strings
def calculate_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# Initialize image descriptions storage
if 'image_descriptions' not in st.session_state:
    st.session_state['image_descriptions'] = []

# Streamlit app layout
st.title("Image Description and Search using GPT-4o")
st.write("Upload an image and get a description using GPT-4o, or search for images related to a text.")

# Textbox for updating OpenAI API key
api_key = st.text_input("Enter your OpenAI API key", type="password")
if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY", "")

if api_key:
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)

    # Section for image uploading and description
    st.header("Upload an Image")
    prompt = st.text_input("Enter the prompt for image description", "What’s in this image?")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")

            # Get the image description
            description = get_image_description(client, uploaded_file, prompt)
            st.write(description)

            # Store the description with the image
            st.session_state['image_descriptions'].append({
                'description': description,
                'image': uploaded_file.getvalue()
            })
        except Exception as e:
            st.error(f"Error: {e}")

    # Section for searching images based on a text input
    st.header("Search for Related Images")
    search_text = st.text_input("Enter text to search for related images")

    if search_text:
        # Calculate similarity of search_text with each stored description
        results = [
            {
                'similarity': calculate_similarity(search_text, item['description']),
                'image': item['image'],
                'description': item['description']
            }
            for item in st.session_state['image_descriptions']
        ]

        # Sort results by similarity
        results = sorted(results, key=lambda x: x['similarity'], reverse=True)

        # Display results
        st.write("Search Results:")
        for result in results:
            st.image(result['image'], caption=f"Description: {result['description']} (Similarity: {result['similarity']:.2f})", use_column_width=True)

else:
    st.error("Please provide a valid OpenAI API key.")

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
