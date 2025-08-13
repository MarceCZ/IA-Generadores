import whisper
from moviepy import VideoFileClip
import os
import torch

# Configura aquí las variables
MODELO = "large"         
RUTA_ARCHIVO = "C:/Users/IALAB/Downloads/El VERDADERO SIGNIFICADO de los SUEÑOS _ Inception _ Relato y Reflexiones.mp4"  
ARCH_SALIDA = "transcripcion.txt"

def es_video_o_audio(ruta_archivo):
    extensiones_video = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv']
    extensiones_audio = ['.mp3', '.wav', '.aac', '.flac', '.ogg', '.m4a']

    ext = os.path.splitext(ruta_archivo)[1].lower()

    if ext in extensiones_video:
        try:
            clip = VideoFileClip(ruta_archivo)
            clip.close()
            return "video"
        except Exception:
            pass

    if ext in extensiones_audio:
        return "audio"

    try:
        clip = VideoFileClip(ruta_archivo)
        clip.close()
        return "video"
    except Exception:
        pass

    return "audio"


def transcribir_archivo(ruta_archivo, modelo):
    tipo = es_video_o_audio(ruta_archivo)

    if tipo == "video":
        # Extraer audio temporalmente
        video = VideoFileClip(ruta_archivo)
        audio_path = "temp_audio.wav"
        video.audio.write_audiofile(audio_path, codec='pcm_s16le')
        video.close()
        ruta_audio = audio_path
    else:
        # Ya es audio
        ruta_audio = ruta_archivo

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # Cargar modelo Whisper
    model = whisper.load_model(modelo, device=device)

    # Transcribir
    resultado = model.transcribe(ruta_audio)

    # Borrar archivo temporal si fue video
    if tipo == "video" and os.path.exists(audio_path):
        os.remove(audio_path)

    return resultado["text"]


def guardar_transcripcion(texto, archivo_salida):
    with open(archivo_salida, "w", encoding="utf-8") as f:
        f.write(texto)
    print(f"Transcripción guardada en '{archivo_salida}'")


def main():
    texto = transcribir_archivo(RUTA_ARCHIVO, MODELO)
    guardar_transcripcion(texto, ARCH_SALIDA)


if __name__ == "__main__":
    main()
