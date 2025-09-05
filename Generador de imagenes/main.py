import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
from PIL import Image, ImageTk
import torch

from diffusers import DiffusionPipeline

MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"

def rounded_rectangle(canvas, x1, y1, x2, y2, radius=18, **kwargs):
    points = [
        x1+radius, y1,
        x2-radius, y1,
        x2, y1,
        x2, y1+radius,
        x2, y2-radius,
        x2, y2,
        x2-radius, y2,
        x1+radius, y2,
        x1, y2,
        x1, y2-radius,
        x1, y1+radius,
        x1, y1
    ]
    return canvas.create_polygon(points, smooth=True, **kwargs)

class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command, bg, fg, activebg, width=220, height=40, radius=18, font=("Arial", 12)):
        super().__init__(parent, width=width, height=height, bg=parent["bg"], highlightthickness=0)
        self.command = command
        self.bg = bg
        self.activebg = activebg
        self.radius = radius
        self.width = width
        self.height = height
        self.font = font
        self.text = text
        self.fg = fg
        self.rect = rounded_rectangle(self, 2, 2, width-2, height-2, radius=radius, fill=bg, outline="")
        self.label = self.create_text(width//2, height//2, text=text, fill=fg, font=font)
        # SOLO UN BIND para evitar múltiples ejecuciones del comando
        self.bind("<Button-1>", lambda e: command())
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        self.itemconfig(self.rect, fill=self.activebg)
    def on_leave(self, event):
        self.itemconfig(self.rect, fill=self.bg)
    def config_state(self, state):
        if state == tk.DISABLED:
            self.itemconfig(self.rect, fill="#444c56")
            self.itemconfig(self.label, fill="#888888")
            self.unbind("<Button-1>")
        else:
            self.itemconfig(self.rect, fill=self.bg)
            self.itemconfig(self.label, fill=self.fg)
            self.bind("<Button-1>", lambda e: self.command())

class StableDiffusionApp:
    def __init__(self, master):
        self.master = master
        master.title("Generador de Imágenes")
        master.geometry("800x800")
        master.resizable(False, False)
        master.configure(bg="#24292F")

        # Frame principal
        self.main_frame = tk.Frame(master, bg="#24292F")
        self.main_frame.pack(expand=True)

        # Prompt label
        self.prompt_label = tk.Label(self.main_frame, text="Describe la imagen que quieres generar:",
                                     font=("Arial", 13), bg="#24292F", fg="#F6F8FA")
        self.prompt_label.grid(row=0, column=0, columnspan=2, pady=(30, 5))

        # Canvas para efecto redondeado
        self.entry_canvas = tk.Canvas(self.main_frame, width=380, height=40, bg="#24292F", highlightthickness=0)
        self.entry_canvas.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        rounded_rectangle(self.entry_canvas, 2, 2, 378, 38, radius=18, fill="#363b42", outline="#30363d")

        # Entry encima del canvas
        self.prompt_entry = tk.Entry(self.main_frame, bd=0, bg="#363b42", fg="white", font=("Arial", 12),
                                     insertbackground="#dadada", justify="left")
        self.prompt_entry.place(in_=self.entry_canvas, x=16, y=8, width=348, height=24)
        self.prompt_entry.insert(0, "")

        # Botón para generar (redondeado)
        self.generate_button = RoundedButton(
            self.main_frame, text="Generar Imagen", command=self.start_generation,
            bg="#238636", fg="white", activebg="#2ea043")
        self.generate_button.grid(row=2, column=0, columnspan=2, pady=14)

        # Estado
        self.status_label = tk.Label(self.main_frame, text="", font=("Arial", 12, "italic"),
                                     bg="#24292F", fg="#8B949E")
        self.status_label.grid(row=3, column=0, columnspan=2, pady=10)

        # Imagen (no visible al inicio)
        self.image_label = tk.Label(self.main_frame, bg="#24292F")
        self.image_label.grid(row=4, column=0, columnspan=2, pady=8)
        self.image_label.grid_remove()

        # Botón para guardar (redondeado, no visible al inicio)
        self.save_button = RoundedButton(
            self.main_frame, text="Guardar Imagen", command=self.save_image,
            bg="#0969DA", fg="white", activebg="#218bff")
        self.save_button.grid(row=5, column=0, columnspan=2, pady=10)
        self.save_button.grid_remove()

        self.generated_image = None
        self.pipe = None
        self.last_save_path = None  # atributo para recordar la última ruta

    def start_generation(self):
        prompt = self.prompt_entry.get().strip()
        if not prompt:
            messagebox.showerror("Error", "Introduce un prompt para generar la imagen.")
            return
        self.status_label.config(text="Generando imagen, por favor espera...")
        self.generate_button.config_state(tk.DISABLED)
        self.prompt_entry.config(state=tk.DISABLED)
        self.save_button.grid_remove()
        self.image_label.grid_remove()
        self.last_save_path = None  # al generar una nueva imagen, borra la ruta
        Thread(target=self.generate_image, args=(prompt,), daemon=True).start()

    def generate_image(self, prompt):
        try:
            if self.pipe is None:
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self.pipe = DiffusionPipeline.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float16 if device=="cuda" else torch.float32,
                    use_safetensors=True,
                    variant="fp16" if device=="cuda" else None
                )
                self.pipe = self.pipe.to(device)
            image = self.pipe(prompt=prompt).images[0]
            self.generated_image = image
            imgtk = ImageTk.PhotoImage(image.resize((420, 420)))
            self.master.after(0, self.display_image, imgtk)
        except Exception as e:
            self.master.after(0, lambda: messagebox.showerror("Error", str(e)))
            self.master.after(0, self.reset_ui)

    def display_image(self, imgtk):
        self.image_label.imgtk = imgtk
        self.image_label.config(image=imgtk)
        self.image_label.grid()
        self.status_label.config(text="Imagen generada.")
        self.save_button.grid()
        self.generate_button.config_state(tk.NORMAL)
        self.prompt_entry.config(state=tk.NORMAL)

    def save_image(self):
        if self.generated_image is None:
            return
        # Si ya hay ruta previa, guarda directo
        if self.last_save_path:
            self.generated_image.save(self.last_save_path)
            messagebox.showinfo("Guardado", f"Imagen guardada en {self.last_save_path}")
            return
        # Si no hay ruta, pide al usuario
        file_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("All files", "*.*")]
        )
        if file_path:
            self.generated_image.save(file_path)
            self.last_save_path = file_path
            messagebox.showinfo("Guardado", f"Imagen guardada en {file_path}")

    def reset_ui(self):
        self.status_label.config(text="")
        self.generate_button.config_state(tk.NORMAL)
        self.prompt_entry.config(state=tk.NORMAL)
        self.save_button.grid_remove()
        self.image_label.grid_remove()
        self.last_save_path = None  # también limpiar al resetear

if __name__ == "__main__":
    root = tk.Tk()
    app = StableDiffusionApp(root)
    root.mainloop()