from fastapi import FastAPI, File, UploadFile
from src.model.tintora_ai import TintoraAI
from src.model.preprocess import preprocess_image
from src.model.postprocess import postprocess_image
import torch
from PIL import Image

app = FastAPI(title="TintoraAI API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
def load_model():
    model = TintoraAI().to(device)
    model.eval()
    return model

@app.post("/colorize")
async def colorize_image(file: UploadFile = File(...), saturation: float = 1.0):
    image = Image.open(file.file)
    input_tensor, original_size = preprocess_image(image)
    input_tensor = input_tensor.to(device)
    model = load_model()
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
    
    colored_image = postprocess_image(output_tensor, original_size, saturation)
    output_file = "api_output.jpg"
    colored_image.save(output_file)
    return {"filename": output_file}
