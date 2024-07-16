import os
import streamlit as st
from openai import OpenAI
import base64

# Streamlit app layout
st.title("Image Description using GPT-4o")
st.write("Upload images and get descriptions using GPT-4o.")

# Textbox for updating OpenAI API key
api_key = st.text_input("Enter your OpenAI API key", type="password")
if not api_key:
    api_key = os.environ.get("OPENAI_API_KEY", "")

if api_key:
    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)

    # Textbox for updating the prompt
    prompt = st.text_input("Enter the prompt for image description", "What’s in this image?")

    # Upload multiple images button
    uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                # Display the uploaded image
                st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)
                st.write("")
                st.write("Classifying...")

                # Get the image description
                description = get_image_description(client, uploaded_file, prompt)
                st.write(description)
                st.write("---")  # Separator between images
            except Exception as e:
                st.error(f"Error: {e}")
else:
    st.error("Please provide a valid OpenAI API key.")

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
