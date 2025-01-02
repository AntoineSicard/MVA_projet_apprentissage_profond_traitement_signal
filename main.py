# %%

#Import des libairies
import scipy
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

import torch

import matplotlib.pyplot as plt
import io
from PIL import Image
import scipy.io.wavfile
import IPython.display as ipd
import pytorch_lightning as pl
from torch.utils.data import random_split
from torch.utils.data import Dataset,DataLoader
from model.waveunet import Waveunet
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# %%
import subprocess
import time

# %%
import torch
import torch.nn as nn

# %% [markdown]
# # Analyse des datas

# %%
#Foncion pour lire les 3 fichiers audio
def read_audio_folder(numero_fichier="0000",chemin_fichier=r"data_source_separation\train_small\train_small"):
    chemin = chemin_fichier + "\\" + numero_fichier
    
    for fichier in os.listdir(chemin):
        chemin_fichier = chemin + "\\" + fichier
        if "mix" in fichier:
            frequence_mix, signal_mix = scipy.io.wavfile.read(chemin_fichier)
        elif "noise" in fichier:
            frequence_noise, signal_noise = scipy.io.wavfile.read(chemin_fichier)
        elif "voice" in fichier:
            frequence_voice, signal_voice = scipy.io.wavfile.read(chemin_fichier)
        else:
            print("Fichier non reconnu")
    if frequence_mix != frequence_noise or frequence_mix != frequence_voice:
        assert False, "Les fréquences des fichiers ne sont pas les mêmes"
    return {"frequence_mix":frequence_mix, "signal_mix":signal_mix, "frequence_noise":frequence_noise, "signal_noise":signal_noise, "frequence_voice":frequence_voice, "signal_voice":signal_voice}


def plot_audio_folder(numero_fichier="0000",chemin_fichier=r"data_source_separation\train_small\train_small"):
    #Read one audio file
    audio = read_audio_folder(numero_fichier,chemin_fichier)

    type_fichier = (
    "train" if "train" in chemin_fichier 
    else "test" if "test" in chemin_fichier 
    else "train_small" if "train_small" in chemin_fichier 
    else "unknown"
)

    # Visualiser les fichiers audio
    figure, axes = plt.subplots(1, 4, figsize=(20, 6))
    figure.suptitle(f"Signaux du fichier {numero_fichier} de type {type_fichier}")
    # Signal mix
    axes[0].plot(audio["signal_mix"])
    axes[0].set_title("Signal mix")
    ipd.display(ipd.Audio(data=audio["signal_mix"], rate=audio["frequence_mix"]))
    sns.despine()

    # Signal noise
    axes[1].plot(audio["signal_noise"])
    axes[1].set_title("Signal noise")
    ipd.display(ipd.Audio(data=audio["signal_noise"], rate=audio["frequence_noise"]))
    sns.despine()

    # Signal voice
    axes[2].plot(audio["signal_voice"])
    axes[2].set_title("Signal voice")
    ipd.display(ipd.Audio(data=audio["signal_voice"], rate=audio["frequence_voice"]))
    sns.despine()

    #Le tout
    axes[3].plot(audio["signal_mix"], label="mix", alpha=0.5)
    axes[3].plot(audio["signal_noise"], label="noise", alpha=0.5)
    axes[3].plot(audio["signal_voice"], label="voice", alpha=0.5)
    axes[3].legend()
    axes[3].set_title("Superposition des signaux")
    sns.despine()


    plt.show()
    



# %%
class Dataset_audio(Dataset):
    
    def __init__(self,data_dir):
        self.data_dir = data_dir
        self.names =[name_folder for name_folder in os.listdir(data_dir)]

        
    def __len__(self): # retourne le nombre de données dans le dataset
        return len(self.names)
    
    def __getitem__(self, i):
        audio = read_audio_folder(numero_fichier=self.names[i],chemin_fichier=self.data_dir)

        signal_mix_i = torch.tensor(audio["signal_mix"], dtype=torch.float32).unsqueeze(0)
        signal_noise_i = torch.tensor(audio["signal_noise"], dtype=torch.float32).unsqueeze(0)
        signal_voice_i = torch.tensor(audio["signal_voice"], dtype=torch.float32).unsqueeze(0)
        frequence_i = torch.tensor(audio["frequence_mix"], dtype=torch.long)

        return signal_mix_i, signal_noise_i, signal_voice_i,frequence_i

# %%
# data = Dataset_audio(parent_folder=r"data_source_separation\train_small\train_small")
# print(len(data))
# print(data[0][0].shape)
# print(data[0][1].shape)
# print(data[0][2].shape)
# print(data[0][3].shape) 

# %%


# %%
class AudioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir,
        test_dir,
        batch_size=32,
        num_workers=4,
        val_split=0.2,
    ):
        super().__init__()
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split


    def setup(self, stage=None):
        if stage == "fit":
            full_dataset = Dataset_audio(
                data_dir=self.train_dir,
            )

            total_size = len(full_dataset)
            val_size = int(total_size * self.val_split)
            train_size = total_size - val_size

            self.train_dataset, self.val_dataset = random_split(
                full_dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),  # For reproducibility
            )

        elif stage == "predict":
            self.predict_dataset = Dataset_audio(
                data_dir=self.test_dir,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=True if self.num_workers > 0 else False,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pertistent_workers=True if self.num_workers > 0 else False,
            shuffle=False,
        )

# %%

# %%


class UNetAudio(nn.Module):
    def __init__(self, input_channels=1, output_channels=2):
        super(UNetAudio, self).__init__()
        # Exemple simple de U-Net avec deux sorties (voix et bruit)
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.middle = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv1d(64, output_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x  # Sorties : [B, 2, T] (voix et bruit)


# %%
class JointLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(JointLoss, self).__init__()
        self.alpha = alpha
        self.l1_loss = nn.L1Loss()

    def forward(self, prediction_voice, signal_voice, prediction_noise, signal_noise):
        # Perte pour la composante voix
        voice_loss = self.l1_loss(prediction_voice, signal_voice)
        
        # Perte pour la composante bruit
        noise_loss = self.l1_loss(prediction_noise, signal_noise)
        
        # Total loss (pondérée)
        total_loss = self.alpha * voice_loss + (1 - self.alpha) * noise_loss
        return total_loss




def plot_signals(signal_mix, signal_voice, signal_noise, prediction_voice, prediction_noise, batch_idx):
    """
    Crée un graphique avec les signaux pour le premier batch.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    
    # Voix : Réel vs Prédiction
    axs[0].plot(signal_voice[0].cpu().numpy(), label="Voix réelle", color="blue")
    axs[0].plot(prediction_voice[0].detach().cpu().numpy(), label="Voix prédite", color="orange", linestyle="--")
    axs[0].set_title("Signal Voix")
    axs[0].legend()
    
    # Bruit : Réel vs Prédiction
    axs[1].plot(signal_noise[0].cpu().numpy(), label="Bruit réel", color="green")
    axs[1].plot(prediction_noise[0].detach().cpu().numpy(), label="Bruit prédit", color="red", linestyle="--")
    axs[1].set_title("Signal Bruit")
    axs[1].legend()
    
    plt.tight_layout()
    
    # Convertir la figure matplotlib en un tableau numpy
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)  # Ferme la figure pour libérer de la mémoire
    return img


# %%
class MainModelLightningModule(pl.LightningModule):
    def __init__(
        self,
        model_str="unet",
        learning_rate=1e-3,
        max_epochs=100,
        optimizer_type="adamw",
        device="cuda",
        lr_scheduler="cosineannealing",
    ):
        super().__init__()
        self.save_hyperparameters()

        if self.hparams.model_str == "waveunet":
            pass
        elif self.hparams.model_str == "unet": 
            self.model=UNetAudio(input_channels=1, output_channels=2)
        # Loss functions
        self.loss = JointLoss(alpha=0.5)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        signal_mix, signal_noise, signal_voice, frequence = batch
        predictions = self(signal_mix)
        prediction_voice = predictions[:, 0, :]  # Voix
        prediction_noise = predictions[:, 1, :]  # Bruit
        
        # Calcul de la perte
        loss = self.loss(prediction_voice, signal_voice.squeeze(), prediction_noise, signal_noise.squeeze())
        self.log("train_loss", loss)

        # Log the learning rate
        lr = self.optimizers().param_groups[0][
            "lr"
        ]  # Récupérer le learning rate actuel
        self.log("learning_rate", lr)  # Enregistrer le learning rate

        return loss

    def validation_step(self, batch, batch_idx):
        signal_mix, signal_noise, signal_voice, frequence = batch
        predictions = self(signal_mix)
        # Sépare les prédictions pour voix et bruit
        prediction_voice = predictions[:, 0, :]  # Voix
        prediction_noise = predictions[:, 1, :]  # Bruit
        
        # Calcul de la perte
        loss = self.loss(prediction_voice, signal_voice.squeeze(), prediction_noise, signal_noise.squeeze())        

        self.validation_step_loss.append(loss)

        # Log graphique pour le premier batch
        # if batch_idx == 0:  # Seulement pour le premier batch
        #     image = plot_signals(signal_mix, signal_voice, signal_noise, prediction_voice, prediction_noise, batch_idx)
        #     self.logger.experiment.add_image("Validation/Signals", image, self.current_epoch)

    def on_validation_epoch_start(
        self,
    ):
        # Réinitialiser les listes au début de chaque époque de validation
        self.validation_step_loss = []

    def on_validation_epoch_end(self):
        # Calculate average loss and dice score over all validation batches
        avg_loss = torch.mean(torch.stack(self.validation_step_loss))

        # Log the average metrics
        self.log("metrics/val_loss_epoch", avg_loss, prog_bar=True)

        self.validation_step_loss.clear()

    # def predict_step(self, batch, batch_idx, dataloader_idx=None):
    #     images, masks, image_files, original_shapes = batch
    #     logits = self(images)
    #     preds = torch.argmax(logits, dim=1)

    #     # Inverse resize des prédictions
    #     preds_resized = []
    #     for pred, original_shape in zip(preds, original_shapes):
    #         pred_resized = nn.functional.interpolate(
    #             pred.unsqueeze(0).float(),  # Ajouter une dimension pour le batch
    #             size=original_shape[1:],  # Taille d'origine (H, W)
    #             mode="nearest",  # Mode d'interpolation
    #             align_corners=False,
    #         )
    #         preds_resized.append(
    #             pred_resized.squeeze(0).cpu().numpy()
    #         )  # Retirer la dimension du batch

    #     return {
    #         "predictions": preds_resized,
    #         "image_files": image_files,
    #         "original_shapes": original_shapes,
    #     }

    def configure_optimizers(self):
        # Choose optimizer
        if self.hparams.optimizer_type.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), lr=self.hparams.learning_rate
            )
        elif self.hparams.optimizer_type.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.hparams.learning_rate
            )
        else:
            raise ValueError("Unsupported optimizer type. Choose 'adam' or 'adamw'.")

        # Cosine Annealing Scheduler
        if self.hparams.lr_scheduler.lower() == "cosineannealing":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.hparams.max_epochs, eta_min=1e-6
            )  # T_max is the number of epochs
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "metrics/val_dice_epoch",
                },
            }

        else:
            return {"optimizer": optimizer}


# %%


# %%


# %%
def launch_tensorboard(
    logdir,
    port=6006,
    path_tensorboard=r"C:\Users\antoi\anaconda3\envs\deep_learning\Lib\site-packages\tensorboard\main.py",
    python_executable=r"C:\Users\antoi\anaconda3\envs\deep_learning\python.exe"
):
    """Launch TensorBoard on the specified port."""
    # Kill any existing TensorBoard instances
    subprocess.run("taskkill /IM tensorboard.exe /F", shell=True, stderr=subprocess.DEVNULL)

    # Launch TensorBoard with the specified port and allow external access
    cmd = f"{python_executable} {path_tensorboard} --logdir={logdir} --port={port}"
    subprocess.Popen(cmd, shell=True)

    # Wait for TensorBoard to start
    time.sleep(3)

    print(f"\nTensorBoard is running on port {port}")
    print(f"Open in your browser:")
    print(f"   http://localhost:{port}")


# %%
def main():    
    for i in range(0):
        plot_audio_folder(numero_fichier=f"{i:04d}")

    from datetime import datetime

    logger = TensorBoardLogger(
        "tb_logs", name="unetaudio", version=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )


    path_tensorboard = r"C:\Users\antoi\anaconda3\envs\apprentissage_profond_traitement_du_signal\Lib\site-packages\tensorboard\main.py"
    python_executable = r"C:\Users\antoi\anaconda3\envs\apprentissage_profond_traitement_du_signal\python.exe"

    launch_tensorboard(
        logdir="tb_logs",
        path_tensorboard=path_tensorboard,
        python_executable=python_executable,
    )


    max_epochs = 100

    model = MainModelLightningModule()
    datamodule = AudioDataModule(train_dir=r"data_source_separation\train_small\train_small", test_dir=r"data_source_separation\test", batch_size=32, num_workers=0, val_split=0.2)

    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.callbacks import LearningRateMonitor

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        callbacks=[
            ModelCheckpoint(
                monitor="metrics/val_loss_epoch",
                mode="min",
                save_top_k=1,
                save_last=True,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        # num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule)
    print('coucou')

    test_results = trainer.predict(model, datamodule, return_predictions=True)




# %%
if __name__ == "__main__":
    main()
    print('cpucpu')