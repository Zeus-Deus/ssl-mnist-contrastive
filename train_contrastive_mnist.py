"""
Lab 4: Self-Supervised Learning - Contrastive Learning
Opdracht 2: Contrastive Learning (Computer Vision)

Doel: Bouw een beeldmodel dat consistente representaties leert door 
verschillend
Dataset: MNIST
Taak: Creëer varianten van elke afbeelding (bijv. door rotatie) en 
train het model om deze te herkennen als dezelfde.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


# ============================================================================
# 1. Device Setup
# ============================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}\n')


# ============================================================================
# 2. Contrastive Transform (Augmentaties)
# ============================================================================
class ContrastiveTransform:
    """Maakt twee verschillende augmentaties van dezelfde afbeelding"""
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ])
    
    def __call__(self, x):
        # Maak twee verschillende augmentaties van dezelfde afbeelding
        return self.transform(x), self.transform(x)


# ============================================================================
# 3. Dataset Laden
# ============================================================================
print("Dataset laden...")
train_dataset = datasets.MNIST(root='./data', train=True, download=True, 
                               transform=ContrastiveTransform())
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

test_dataset = datasets.MNIST(root='./data', train=False, download=True,
                              transform=transforms.ToTensor())
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

print(f'Training dataset size: {len(train_dataset)}')
print(f'Test dataset size: {len(test_dataset)}\n')


# ============================================================================
# 4. Model Definitie
# ============================================================================
class Encoder(nn.Module):
    """CNN Encoder voor het leren van representaties"""
    def __init__(self, embedding_dim=128):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # Normaliseer embeddings
        return nn.functional.normalize(x, dim=1)


model = Encoder().to(device)
print("Model architectuur:")
print(model)
print()


# ============================================================================
# 5. Contrastive Loss (NT-Xent Loss)
# ============================================================================
class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss"""
    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, z_i, z_j):
        batch_size = z_i.shape[0]
        # Concateneer beide views
        z = torch.cat([z_i, z_j], dim=0)
        
        # Bereken similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature
        
        # Haal positieve pairs eruit
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        
        positive_samples = torch.cat([sim_i_j, sim_j_i], dim=0).reshape(batch_size * 2, 1)
        
        # Mask voor negatieve samples (alles behalve diagonaal)
        mask = torch.eye(batch_size * 2, dtype=torch.bool, device=device)
        negative_samples = sim[~mask].reshape(batch_size * 2, -1)
        
        # Combineer positieve en negatieve samples
        logits = torch.cat([positive_samples, negative_samples], dim=1)
        labels = torch.zeros(batch_size * 2, dtype=torch.long, device=device)
        
        loss = nn.functional.cross_entropy(logits, labels)
        return loss


criterion = NTXentLoss().to(device)


# ============================================================================
# 6. Training
# ============================================================================
print("\n" + "="*70)
print("Start training...")
print("="*70 + "\n")

optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 5

losses = []

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for batch_idx, ((x_i, x_j), _) in enumerate(train_loader):
        x_i, x_j = x_i.to(device), x_j.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        z_i = model(x_i)
        z_j = model(x_j)
        
        # Bereken loss
        loss = criterion(z_i, z_j)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Step [{batch_idx}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    avg_loss = epoch_loss / len(train_loader)
    losses.append(avg_loss)
    print(f'\nEpoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')
    print("-"*70 + "\n")

print("Training voltooid!\n")


# ============================================================================
# 7. Training Loss Visualisatie
# ============================================================================
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.title('Training Loss over Epochs')
plt.grid(True, alpha=0.3)
plt.savefig('training_loss.png', dpi=150, bbox_inches='tight')
print("✓ Training loss plot opgeslagen: training_loss.png")


# ============================================================================
# 8. Evaluatie: Embeddings Genereren
# ============================================================================
print("\nGenereren van embeddings...")
model.eval()
embeddings = []
labels = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        z = model(x)
        embeddings.append(z.cpu().numpy())
        labels.append(y.numpy())
        break  # Neem alleen eerste batch voor snelheid

embeddings = np.concatenate(embeddings, axis=0)
labels = np.concatenate(labels, axis=0)

print(f'Embeddings shape: {embeddings.shape}')
print(f'Labels shape: {labels.shape}')


# ============================================================================
# 9. t-SNE Visualisatie
# ============================================================================
print("\nBezig met t-SNE visualisatie...")
tsne = TSNE(n_components=2, random_state=42, verbose=1)
embeddings_2d = tsne.fit_transform(embeddings)

plt.figure(figsize=(12, 10))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     c=labels, cmap='tab10', alpha=0.6, s=20)
plt.colorbar(scatter, label='Digit Class')
plt.title('t-SNE Visualisatie van Geleerde Representaties', fontsize=14)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True, alpha=0.3)
plt.savefig('tsne_visualization.png', dpi=150, bbox_inches='tight')
print("✓ t-SNE visualisatie opgeslagen: tsne_visualization.png")


# ============================================================================
# 10. Model Opslaan
# ============================================================================
torch.save(model.state_dict(), 'contrastive_model.pth')
print("\n✓ Model opgeslagen: contrastive_model.pth")


# ============================================================================
# Conclusie
# ============================================================================
print("\n" + "="*70)
print("CONCLUSIE")
print("="*70)
print("""
Deze implementatie bevat:
1. ✓ Pretext Task: Verschillende augmentaties van dezelfde afbeelding herkennen
2. ✓ Model: CNN encoder met NT-Xent loss voor contrastive learning
3. ✓ Training: 5 epochs met loss tracking
4. ✓ Evaluatie: t-SNE visualisatie van geleerde representaties

Vergelijkbare afbeeldingen (met dezelfde label) zouden gegroepeerd moeten 
zijn in de t-SNE visualisatie.

Mogelijke verbeteringen:
- Meer epochs trainen (10-20)
- Complexere augmentaties (color jitter, blur, etc.)
- Groter model gebruiken (ResNet)
- Temperature parameter optimaliseren
- Linear evaluation op downstream task
""")
print("="*70)

plt.show()
