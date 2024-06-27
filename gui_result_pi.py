import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageFilter   # Add this import
import os  # Add this import
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from dataset import PhotoSketchDataset
from generator_model import Generator
from discriminator_model import Discriminator
from utils import save_checkpoint, load_checkpoint
import config



class CycleGANTrainer:
    def __init__(self, root):
        self.root = root
        self.root.title("CycleGAN Photo to Sketch Matcher")

        self.create_widgets()
        self.setup_model()

    def create_widgets(self):
        # Dataset frame
        dataset_frame = tk.LabelFrame(self.root, text="Datasets", padx=10, pady=10)
        dataset_frame.pack(padx=10, pady=10, fill="x")

        self.val_dir_label = tk.Label(dataset_frame, text="Input Image Directory:")
        self.val_dir_label.grid(row=1, column=0)
        self.val_dir_entry = tk.Entry(dataset_frame, width=50)
        self.val_dir_entry.grid(row=1, column=1)
        self.val_dir_button = tk.Button(dataset_frame, text="Browse", command=self.browse_val_dir)
        self.val_dir_button.grid(row=1, column=2)

        # Hyperparameters frame
        hyperparams_frame = tk.LabelFrame(self.root, text="Hyperparameters", padx=10, pady=10)
        hyperparams_frame.pack(padx=10, pady=10, fill="x")

        self.lr_label = tk.Label(hyperparams_frame, text="Learning Rate:")
        self.lr_label.grid(row=0, column=0)
        self.lr_entry = tk.Entry(hyperparams_frame, width=10)
        self.lr_entry.insert(tk.END, "1e-5")
        self.lr_entry.grid(row=0, column=1)

        self.batch_size_label = tk.Label(hyperparams_frame, text="Batch Size:")
        self.batch_size_label.grid(row=1, column=0)
        self.batch_size_entry = tk.Entry(hyperparams_frame, width=10)
        self.batch_size_entry.insert(tk.END, "1")
        self.batch_size_entry.grid(row=1, column=1)

        self.epochs_label = tk.Label(hyperparams_frame, text="Number of Epochs:")
        self.epochs_label.grid(row=2, column=0)
        self.epochs_entry = tk.Entry(hyperparams_frame, width=10)
        self.epochs_entry.insert(tk.END, "1")
        self.epochs_entry.grid(row=2, column=1)

        # Control buttons frame
        control_frame = tk.Frame(self.root, padx=10, pady=10)
        control_frame.pack(padx=10, pady=10, fill="x")

        self.start_button = tk.Button(control_frame, text="Start Generating", command=self.start_training)
        self.start_button.pack()

        # Progress frame
        progress_frame = tk.Frame(self.root, padx=10, pady=10)
        progress_frame.pack(padx=10, pady=10, fill="x")

        self.progress = tk.StringVar()
        self.progress.set("Progress: Not started")
        self.progress_label = tk.Label(progress_frame, textvariable=self.progress)
        self.progress_label.pack()


    def browse_val_dir(self):
        val_dir = filedialog.askdirectory()
        if val_dir:
            self.val_dir_entry.insert(0, val_dir)

    def setup_model(self):
        self.disc_H = Discriminator(in_channels=3).to(config.DEVICE)
        self.disc_Z = Discriminator(in_channels=3).to(config.DEVICE)
        self.gen_Z = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
        self.gen_H = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)

        self.opt_disc = optim.Adam(
            list(self.disc_H.parameters()) + list(self.disc_Z.parameters()),
            lr=float(self.lr_entry.get()),
            betas=(0.5, 0.999),
        )

        self.opt_gen = optim.Adam(
            list(self.gen_Z.parameters()) + list(self.gen_H.parameters()),
            lr=float(self.lr_entry.get()),
            betas=(0.5, 0.999),
        )

        self.L1 = nn.L1Loss()
        self.mse = nn.MSELoss()

        if config.LOAD_MODEL:
            load_checkpoint(config.CHECKPOINT_GEN_H, self.gen_H, self.opt_gen, config.LEARNING_RATE)
            load_checkpoint(config.CHECKPOINT_GEN_Z, self.gen_Z, self.opt_gen, config.LEARNING_RATE)
            load_checkpoint(config.CHECKPOINT_CRITIC_H, self.disc_H, self.opt_disc, config.LEARNING_RATE)
            load_checkpoint(config.CHECKPOINT_CRITIC_Z, self.disc_Z, self.opt_disc, config.LEARNING_RATE)

    def start_training(self):
        train_dir = "C:\\Users\\DeLL\\Desktop\\FYP\\CycleGAN\\cycleGAN_trial3\\data\\train"
        val_dir = self.val_dir_entry.get()
        batch_size = int(self.batch_size_entry.get())
        epochs = int(self.epochs_entry.get())

        if not val_dir:
            messagebox.showerror("Error", "Please specify the validation directory.")
            return

        dataset = PhotoSketchDataset(
            root_photo=f"{train_dir}/photos",
            root_sketch=f"{train_dir}/sketches",
            transform=config.transforms,
        )
        val_dataset = PhotoSketchDataset(
            root_photo=f"{val_dir}/photos",
            root_sketch=f"{val_dir}/sketches",
            transform=config.transforms,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=False,
        )
        #loader = DataLoader(
         #   dataset,
          #  batch_size=batch_size,
           # shuffle=True,
            #num_workers=config.NUM_WORKERS,
            #pin_memory=True,
        #)
        g_scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()

        for epoch in range(epochs):
            self.train_fn(
                self.disc_H,
                self.disc_Z,
                self.gen_Z,
                self.gen_H,
                val_loader,
                self.opt_disc,
                self.opt_gen,
                self.L1,
                self.mse,
                d_scaler,
                g_scaler,
            )
            self.progress.set(f"Progress: Epoch {epoch + 1}/{epochs} completed")

            if config.SAVE_MODEL:
                save_checkpoint(self.gen_H, self.opt_gen, filename=config.CHECKPOINT_GEN_H)
                save_checkpoint(self.gen_Z, self.opt_gen, filename=config.CHECKPOINT_GEN_Z)
                save_checkpoint(self.disc_H, self.opt_disc, filename=config.CHECKPOINT_CRITIC_H)
                save_checkpoint(self.disc_Z, self.opt_disc, filename=config.CHECKPOINT_CRITIC_Z)

        # Call show_results after training is completed
        self.show_results()

        messagebox.showinfo("Info", "Successfully generated!")

    def train_fn(
            self, disc_H, disc_Z, gen_Z, gen_H, val_loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
    ):
        H_reals = 0
        H_fakes = 0
        loop = tqdm(val_loader, leave=True)

        for idx, (sketch, photo) in enumerate(loop):
            sketch = sketch.to(config.DEVICE)
            photo = photo.to(config.DEVICE)

            # Train Discriminators H and Z
            with torch.cuda.amp.autocast():
                fake_photo = gen_H(sketch)
                D_H_real = disc_H(photo)
                D_H_fake = disc_H(fake_photo.detach())
                H_reals += D_H_real.mean().item()
                H_fakes += D_H_fake.mean().item()
                D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
                D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
                D_H_loss = D_H_real_loss + D_H_fake_loss

                fake_sketch = gen_Z(photo)
                D_Z_real = disc_Z(sketch)
                D_Z_fake = disc_Z(fake_sketch.detach())
                D_Z_real_loss = mse(D_Z_real, torch.ones_like(D_Z_real))
                D_Z_fake_loss = mse(D_Z_fake, torch.zeros_like(D_Z_fake))
                D_Z_loss = D_Z_real_loss + D_Z_fake_loss

                # put it together
                D_loss = (D_H_loss + D_Z_loss) / 2

            opt_disc.zero_grad()
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()

            # Train Generators H and Z
            with torch.cuda.amp.autocast():
                # adversarial loss for both generators
                D_H_fake = disc_H(fake_photo)
                D_Z_fake = disc_Z(fake_sketch)
                loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))
                loss_G_Z = mse(D_Z_fake, torch.ones_like(D_Z_fake))

                # cycle loss
                #cycle_sketch = gen_Z(fake_photo)
                #cycle_photo = gen_H(fake_sketch)
                #cycle_sketch_loss = l1(sketch, cycle_sketch)
                #cycle_photo_loss = l1(photo, cycle_photo)

                # add all together
                G_loss = (
                    loss_G_Z
                    + loss_G_H
                    #+ cycle_sketch_loss * config.LAMBDA_CYCLE
                    #+ cycle_photo_loss * config.LAMBDA_CYCLE
                )

            opt_gen.zero_grad()
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()

            if idx % 200 == 0:
                save_image(fake_photo * 0.5 + 0.5, f"saved_images/photo_{idx}.png")
                save_image(fake_sketch * 0.5 + 0.5, f"saved_images/sketch_{idx}.png")

            loop.set_postfix(H_real=H_reals / (idx + 1), H_fake=H_fakes / (idx + 1))

    def show_results(self):
        result_window = tk.Toplevel(self.root)
        result_window.title("Generated Images")

        photo_frame = tk.LabelFrame(result_window, text="Generated Photo and Original Photo", padx=10, pady=10)
        photo_frame.pack(side="left", padx=20, pady=20)

        sketch_frame = tk.LabelFrame(result_window, text="Generated Sketch and Hand Drawn Sketch", padx=10, pady=10)
        sketch_frame.pack(side="right", padx=20, pady=20)

        # Load and display generated photos
        for i in range(1):
            photo_path = f"saved_images/photo_0.png"
            photo = tk.PhotoImage(file=photo_path)
            photo_label = tk.Label(photo_frame, image=photo)
            photo_label.image = photo
            photo_label.pack()

        # Load and display generated sketches
        for i in range(1):
            sketch_path = f"saved_images/sketch_0.png"
            sketch = tk.PhotoImage(file=sketch_path)
            sketch_label = tk.Label(sketch_frame, image=sketch)
            sketch_label.image = sketch
            sketch_label.pack()

        val_dir = self.val_dir_entry.get()

        if not val_dir:
            messagebox.showerror("Error", "Please specify the validation directory.")
            return

        photo_files = sorted([f for f in os.listdir(os.path.join(val_dir, 'photos')) if f.endswith('.jpg')])
        sketch_files = sorted([f for f in os.listdir(os.path.join(val_dir, 'sketches')) if f.endswith('.jpg')])

        for photo_file, sketch_file in zip(photo_files, sketch_files):
            photo_path = os.path.join(val_dir, 'photos', photo_file)
            sketch_path = os.path.join(val_dir, 'sketches', sketch_file)

            photo_image = Image.open(photo_path)
            photo_image = photo_image.resize((128, 128), Image.BILINEAR)
            photo = ImageTk.PhotoImage(photo_image)
            photo_label = tk.Label(photo_frame, image=photo)
            photo_label.image = photo
            photo_label.pack()

            sketch_image = Image.open(sketch_path)
            sketch_image = sketch_image.resize((128, 128), Image.BILINEAR)
            sketch = ImageTk.PhotoImage(sketch_image)
            sketch_label = tk.Label(sketch_frame, image=sketch)
            sketch_label.image = sketch
            sketch_label.pack()

if __name__ == "__main__":
    root = tk.Tk()
    app = CycleGANTrainer(root)
    root.mainloop()
