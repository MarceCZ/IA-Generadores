from diffusers import DiffusionPipeline
import torch
from PIL import Image

def main():

    model_name = "sd-legacy/stable-diffusion-v1-5"

    # Cargar modelo (se descargará la primera vez)
    pipe = DiffusionPipeline.from_pretrained(
        model_name, 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        variant="fp16"
    )

    # Enviar modelo a GPU si está disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    pipe = pipe.to(device)

    # Definir prompt (lo que quieres generar)
    prompt = "familia de 4 integrantes en una tarde de picnic"

    # Generar imagen
    images = pipe(prompt=prompt).images[0]

    # Guardar imagen
    images.show()
    images.save("./picture/img1.png")

if __name__ == "__main__":
    main()
