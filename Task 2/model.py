import torch

class Conv1DFrontEnd(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=11, stride=5):
        super().__init__()
        self.conv1d = torch.nn.Conv1d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        # x shape: (batch_size, 1, time)
        return self.conv1d(x)
    

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # shape (1, max_len, d_model)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x



class AudioTransformer(torch.nn.Module):
    def __init__(self, input_dim=64, nhead=4, num_layers=3):
        super().__init__()
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead)
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (batch_size, channels, seq_len)
        x = x.permute(2, 0, 1)  # shape: (seq_len, batch_size, channels)
        return self.transformer(x)



class ClassifierHead(torch.nn.Module):
    def __init__(self, input_dim=64, num_classes=10):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = x.permute(1, 2, 0)  # (batch, channels, seq_len)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class SoundClassifier(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.frontend = Conv1DFrontEnd()
        self.transformer = AudioTransformer()
        self.classifier = ClassifierHead()

    def forward(self, x):
        x = self.frontend(x)
        x = self.transformer(x)
        return self.classifier(x)
