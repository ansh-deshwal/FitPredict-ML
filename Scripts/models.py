"""Neural network models"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class BaselineMLP(nn.Module):
    """Simple MLP baseline"""
    def __init__(self, input_dim=1280):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        return self.network(x)

class ProteinAutoencoder(nn.Module):
    """Autoencoder: 1280 -> 64 -> 1280"""
    def __init__(self, input_dim=1280, latent_dim=64, dropout=0.2):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, input_dim)
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return encoded, decoded

class ContactMapCNN(nn.Module):
    """CNN for contact maps"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.fc = nn.Linear(64 * 64 * 64, 64)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class BiLSTMSequentialEncoder(nn.Module):
    """BiLSTM for sequences"""
    def __init__(self, input_dim=64, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            bidirectional=True, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2, 128)
        
    def forward(self, x):
        lstm_out, (h_n, c_n) = self.bilstm(x)
        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)
        return self.fc(combined)

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(dim),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return F.relu(self.residual_function(x) + x)

class MultiModalFusionNetwork(nn.Module):
    """Complete fusion network"""
    def __init__(self, autoencoder, config):
        super().__init__()
        self.config = config
        
        self.autoencoder = autoencoder
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        
        self.bilstm = BiLSTMSequentialEncoder(
            config.BILSTM_CONFIG['input_dim'],
            config.BILSTM_CONFIG['hidden_dim'],
            config.BILSTM_CONFIG['num_layers'],
            config.BILSTM_CONFIG['dropout']
        )
        
        self.contact_cnn = ContactMapCNN()
        
        fusion_input_dim = (
            config.FUSION_CONFIG['seq_features_dim'] + 
            config.FUSION_CONFIG['struct_features_dim']
        )
        
        self.fusion_input = nn.Sequential(
            nn.Linear(fusion_input_dim, config.FUSION_CONFIG['hidden_dim']),
            nn.BatchNorm1d(config.FUSION_CONFIG['hidden_dim']),
            nn.ReLU()
        )
        
        self.res_blocks = nn.ModuleList([
            ResidualBlock(
                config.FUSION_CONFIG['hidden_dim'],
                config.FUSION_CONFIG['dropout']
            )
            for _ in range(config.FUSION_CONFIG['num_residual_blocks'])
        ])
        
        self.predictor = nn.Sequential(
            nn.BatchNorm1d(config.FUSION_CONFIG['hidden_dim']),
            nn.Linear(config.FUSION_CONFIG['hidden_dim'], 128),
            nn.ReLU(),
            nn.Dropout(config.FUSION_CONFIG['dropout']),
            nn.Linear(128, 1)
        )
        
    def prepare_sequence(self, compressed_embeddings, positions):
        batch_size = compressed_embeddings.size(0)
        max_len = self.config.SEQUENCE_CONFIG['max_seq_length']
        feat_dim = compressed_embeddings.size(1)
        
        seq_tensor = torch.zeros(batch_size, max_len, feat_dim, 
                                 device=compressed_embeddings.device)
        
        for i in range(batch_size):
            pos = positions[i].item()
            if 0 <= pos < max_len:
                seq_tensor[i, pos, :] = compressed_embeddings[i]
        
        return seq_tensor
    
    def forward(self, esm2_embeddings, contact_maps, positions):
        with torch.no_grad():
            compressed, _ = self.autoencoder(esm2_embeddings)
        
        seq_data = self.prepare_sequence(compressed, positions)
        seq_features = self.bilstm(seq_data)
        struct_features = self.contact_cnn(contact_maps)
        
        combined = torch.cat([seq_features, struct_features], dim=1)
        x = self.fusion_input(combined)
        
        for res_block in self.res_blocks:
            x = res_block(x)
        
        return self.predictor(x)

def print_model_summary(model, name="Model"):
    """Print model summary"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n{'='*70}")
    print(f"{name} Summary")
    print(f"{'='*70}")
    print(f"Total:     {total:,}")
    print(f"Trainable: {trainable:,}")
    print(f"Frozen:    {total - trainable:,}")
    print(f"{'='*70}\n")