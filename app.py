import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os

st.set_page_config(
    page_title="Alzheimer's Stage Classifier",
    page_icon="🧠",
    layout="centered"
)


CLASSES = ['NonDemented', 'VeryMildDemented',
           'MildDemented', 'ModerateDemented']
MODEL_PATH = 'alzhemeir_classifier.pth'


@st.cache_resource
def load_model():

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

    if os.path.exists(MODEL_PATH):

        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    else:
        st.error(
            f"Model file '{MODEL_PATH}' not found. Please ensure it is in the same directory.")

    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


st.title("🧠 Alzheimer's Disease Stage Classification")
st.write("Upload an MRI brain scan image to predict the classification stage.")


try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")


uploaded_file = st.file_uploader(
    "Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Scan', use_container_width=True)

    st.write("")

    if st.button('Analyze Scan'):
        with st.spinner('Analyzing image features...'):
            try:

                input_tensor = transform(image).unsqueeze(
                    0)

                with torch.no_grad():
                    outputs = model(input_tensor)

                    probabilities = torch.nn.functional.softmax(
                        outputs[0], dim=0)
                    confidence, preds = torch.max(probabilities, 0)

                predicted_class = CLASSES[preds.item()]
                confidence_score = confidence.item() * 100

                st.markdown("---")
                st.subheader("Prediction Result")

                if predicted_class == 'NonDemented':
                    st.success(
                        f"**Classification:** {predicted_class} ({confidence_score:.2f}% confidence)")
                else:
                    st.warning(
                        f"**Classification:** {predicted_class} ({confidence_score:.2f}% confidence)")

                with st.expander("See Detailed Probability Breakdown"):
                    for idx, prob in enumerate(probabilities):
                        st.write(
                            f"**{CLASSES[idx]}:** {prob.item() * 100:.2f}%")

            except Exception as e:
                st.error(f"An error occurred during inference: {e}")
