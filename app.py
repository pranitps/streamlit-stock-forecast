import streamlit as st
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
import io

# DnCNN Model Definition (50 layers, RGB)
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features,
                                kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels,
                                kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out  # Residual learning

# Load model only once
@st.cache_resource
def load_model():
    model = DnCNN(channels=1, num_of_layers=50)
    state_dict = torch.load("net.pth", map_location=torch.device('cpu'))

    # Strip "module." prefix if model was trained using nn.DataParallel
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")  # remove 'module.' prefix
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


model = load_model()

st.title("🌌 Milky Way Noise Reduction with DnCNN")

uploaded_file = st.file_uploader("Upload a Milky Way image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file).convert("L")
    st.image(input_image, caption="Original Image", use_container_width=True)

    if st.button("✨ Denoise Image"):
        with st.spinner("Processing..."):
            img_tensor = ToTensor()(input_image).unsqueeze(0)
            with torch.no_grad():
                output_tensor = model(img_tensor)
                output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
            output_image = ToPILImage()(output_tensor.squeeze())

        st.success("Denoising complete!")
        st.image(output_image, caption="Denoised Image", use_container_width=True)

        # Download button
        buf = io.BytesIO()
        output_image.save(buf, format="JPEG")
        st.download_button(
            label="📥 Download Denoised Image",
            data=buf.getvalue(),
            file_name="milkyway_denoised.jpg",
            mime="image/jpeg"
        )
