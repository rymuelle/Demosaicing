import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import mlflow
import mlflow.pytorch

# Custom imports
from paths import flickr30k
from src.Flickr30kDatasetCorrupt import Flickr30kDatasetCorrupt
from src.CFA_sim import simulate_sparse_wrapper
from arch.NAFNetNoRes import NAFNet

CONFIG = {
    "model_name": "nafnet_baseline",
    "experiment_name": "Flickr30k_Demosaicing",
    "batch_size": 16,
    "lr": 1e-3,
    "sched_end_factor": 1e-1,
    "epochs": 2,
    "seed": 42,
    "num_workers": 12,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "cfa_type": "xtrans",
    "width": 16,
    "middle_blk_num":10,
    "enc_blk_nums":[2, 2, 4, 6],
    "dec_blk_nums":[2, 2, 2, 2],
}

def train():
    generator = torch.Generator().manual_seed(CONFIG["seed"])

    # Dataset
    transforms = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda x: x / 255)
    ])

    dataset = Flickr30kDatasetCorrupt(
        os.path.join(flickr30k, "Images"), 
        os.path.join(flickr30k, "captions.txt"),
        corrupt=lambda x: simulate_sparse_wrapper(x, cfa_type=CONFIG["cfa_type"]),
        transform=transforms
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=CONFIG["batch_size"], shuffle=True, 
                              generator=generator, num_workers=CONFIG["num_workers"])
    val_loader = DataLoader(val_set, batch_size=CONFIG["batch_size"], shuffle=False, 
                            generator=generator, num_workers=CONFIG["num_workers"])

    # Model
    model = NAFNet(img_channel=3, in_channels=3, width=CONFIG["width"], middle_blk_num=CONFIG["middle_blk_num"], 
                   enc_blk_nums=CONFIG["enc_blk_nums"], dec_blk_nums=CONFIG["dec_blk_nums"]).to(CONFIG["device"])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=CONFIG["sched_end_factor"], total_iters=CONFIG["epochs"])
    criterion = nn.L1Loss()

    # MLflow Tracking
    mlflow.set_experiment(CONFIG["experiment_name"])
    
    with mlflow.start_run(run_name=CONFIG["model_name"]):
        # Log Hyperparameters
        mlflow.log_params(CONFIG)
        
        for epoch in range(CONFIG["epochs"]):
            # Trainig
            model.train()
            train_loss = 0.0
            tloader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']} [Train]")
            
            for images, sparse, _ in tloader:
                images, sparse = images.to(CONFIG["device"]), sparse.to(CONFIG["device"])
                optimizer.zero_grad()
                with torch.autocast(device_type=CONFIG["device"], dtype=torch.bfloat16):
                    output = model(sparse)
                loss = criterion(output, images)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                tloader.set_postfix({"loss": f"{loss.item():.4e}"})
            
            avg_train_loss = train_loss / len(train_set)
            mlflow.log_metric("train_l1_loss", avg_train_loss, step=epoch)
            scheduler.step()

            # Validation
            model.eval()
            val_loss = 0.0
            vloader = tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")
            
            with torch.no_grad():
                for images, sparse, _ in vloader:
                    images, sparse = images.to(CONFIG["device"]), sparse.to(CONFIG["device"])
                    output = model(sparse)
                    loss = criterion(output, images)
                    val_loss += loss.item() * images.size(0)
            
            avg_val_loss = val_loss / len(val_set)
            mlflow.log_metric("val_l1_loss", avg_val_loss, step=epoch)
            print(f"Epoch {epoch+1}: Val Loss: {avg_val_loss:.4e}")

        # Save Artifacts
        mlflow.pytorch.log_model(model, "model")
        
        # Local save as backup
        torch.save(model.state_dict(), f"{CONFIG['model_name']}_final.pth")

if __name__ == "__main__":
    torch.cuda.empty_cache()
    train()