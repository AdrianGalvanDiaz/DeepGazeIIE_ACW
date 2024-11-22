import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
import numpy as np
from deepgaze_pytorch.training import train
from deepgaze_pytorch.data import (
    create_train_val_datasets,
    FixationMaskTransform,
    verify_dataset
)
from deepgaze_pytorch.deepgaze2e import DeepGazeIIE

def main():
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Configurar rutas
    data_root = "C:/Users/Adrian/Downloads/Data DeepGaze"
    checkpoint_dir = "C:/Users/Adrian/Desktop/deepgaze_pytorch/deepgaze_pytorch/checkpoints"
    
    # Crear transform
    transform = FixationMaskTransform(sparse=False)
    
    # Crear datasets con información de categorías
    print("\nCreando datasets...")
    train_dataset, val_dataset, train_baseline_ll, val_baseline_ll = create_train_val_datasets(
        data_root, 
        transform=transform,
        val_split=0.2
    )

    # Obtener número de categorías del dataset
    num_categories = len(train_dataset.category_to_idx)
    print(f"\nNúmero total de categorías: {num_categories}")
    print("\nCategorías disponibles:")
    for cat_name, cat_idx in train_dataset.category_to_idx.items():
        print(f"  {cat_name}: {cat_idx}")

    print(f"\nBaseline Log-Likelihoods:")
    print(f"Training: {train_baseline_ll:.4f}")
    print(f"Validation: {val_baseline_ll:.4f}")
    
    # Verificar datasets
    verify_dataset(train_dataset, "Training Dataset")
    verify_dataset(val_dataset, "Validation Dataset")
    
    # Preguntar al usuario si desea continuar
    response = input("\n¿Los datos se han cargado correctamente? (y/n): ")
    if response.lower() != 'y':
        print("Entrenamiento cancelado por el usuario.")
        return
    
    # Crear dataloaders
    print("\nCreando dataloaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2,
        shuffle=True,
        num_workers=4,  # Aumentar workers para mejor rendimiento
        pin_memory=True  # Mejora rendimiento en GPU
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2,
        num_workers=4,
        pin_memory=True
    )
    
    # Configuración de entrenamiento
    config = {
        'device': device,
        'accumulation_steps': 4,
        'learning_rate': 1e-4,
        'validation_epochs': 1,
        'minimum_learning_rate': 1e-6,
        'num_categories': num_categories
    }
    
    print("\nConfiguración del entrenamiento:")
    for key, value in config.items():
        print(f"{key}: {value}")
    
    # Inicializar modelo con soporte para categorías
    print("\nInicializando modelo...")
    model = DeepGazeIIE(num_categories=config['num_categories'], pretrained=False)
    
    # Optimizer y scheduler
    # Separar parámetros del modelo base y de la capa de categorías para diferentes learning rates
    category_params = list(model.category_weighting.parameters())
    base_params = [p for n, p in model.named_parameters() 
                  if not n.startswith('category_weighting')]
    
    optimizer = torch.optim.Adam([
        {'params': base_params, 'lr': config['learning_rate']},
        {'params': category_params, 'lr': config['learning_rate'] * 10}  # Learning rate más alto para los pesos de categoría
    ])
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # Iniciar entrenamiento
    print("\nIniciando entrenamiento...")
    try:
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_baseline_ll=train_baseline_ll,  
            val_baseline_ll=val_baseline_ll,      
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_dir=checkpoint_dir,
            device=config['device'],
            validation_epochs=config['validation_epochs'],
            minimum_learning_rate=config['minimum_learning_rate'],
            accumulation_steps=config['accumulation_steps']
        )
        
        # Analizar y guardar pesos finales por categoría
        if hasattr(model, 'category_weighting'):
            weights = model.category_weighting.get_category_weights()
            weights_np = weights.detach().cpu().numpy()
            
            # Guardar pesos
            np.save(f"{checkpoint_dir}/final_category_weights.npy", weights_np)
            
            # Imprimir análisis de pesos
            print("\nAnálisis final de pesos por categoría:")
            cat_names = list(train_dataset.category_to_idx.keys())
            for i, cat_name in enumerate(cat_names):
                print(f"\n{cat_name}:")
                print("Pesos de encoder:", weights_np[i])
                print(f"Encoder dominante: {np.argmax(weights_np[i])}")
                print(f"Peso máximo: {np.max(weights_np[i]):.4f}")
        
    except Exception as e:
        print(f"\nError durante el entrenamiento: {str(e)}")
        print("Traceback completo:")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    main()