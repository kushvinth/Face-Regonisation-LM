def main():
    import torch
    import os
    from src.face_pipeline import SimpleFaceNN, get_data_loaders, train

    # Config
    data_dir = "./data/faces"  # Change this to your dataset path
    batch_size = 16
    num_epochs = 3
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found. Please add your face images.")
        return

    train_loader, val_loader, num_classes = get_data_loaders(data_dir, batch_size)
    model = SimpleFaceNN(num_classes=num_classes).to(device)
    train(model, train_loader, val_loader, device, num_epochs=num_epochs, lr=lr)

if __name__ == "__main__":
    main()
