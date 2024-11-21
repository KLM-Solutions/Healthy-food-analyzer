import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import openai
import io
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load CLIP model and processor
@st.cache_resource  # Cache the model loading
def load_models():
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor

# Define labels for classification
LABELS = ["Clearly Healthy", "Borderline", "Mixed", "Clearly Unhealthy"]

def analyze_food(image):
    clip_model, clip_processor = load_models()
    
    try:
        # Resize image if needed
        max_size = 768
        ratio = min(max_size/image.width, max_size/image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # CLIP processing
        inputs = clip_processor(
            text=LABELS, images=image, return_tensors="pt", padding=True
        )
        
        # Perform classification with CLIP
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        predicted_idx = probs.argmax(dim=1).item()
        predicted_label = LABELS[predicted_idx]
        confidence_score = probs[0][predicted_idx].item() * 100
        
        # Convert image for GPT
        buffered = io.BytesIO()
        resized_image.save(buffered, format="JPEG", quality=80)
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        # GPT Analysis
        gpt_response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Analyze the food image based on health criteria. Be concise."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Category: {predicted_label}\nConfidence: {confidence_score:.2f}%"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=300
        )
        
        gpt_analysis = gpt_response['choices'][0]['message']['content']
        
        return {
            "category": predicted_label,
            "confidence": confidence_score,
            "analysis": gpt_analysis
        }
        
    except Exception as e:
        return {"error": str(e)}

def main():
    st.title("Food Health Analyzer")
    st.write("Upload a food image to analyze its health score and get recommendations.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with st.spinner('Analyzing image...'):
            result = analyze_food(image)
            
            if "error" in result:
                st.error(f"Error analyzing image: {result['error']}")
            else:
                st.subheader("Results:")
                st.write(f"**Category:** {result['category']}")
                st.write(f"**Confidence:** {result['confidence']:.2f}%")
                st.write("**Analysis:**")
                st.write(result['analysis'])

if __name__ == "__main__":
    main()