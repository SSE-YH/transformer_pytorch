import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import math
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*-(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term) # 짝수 인덱스
        pe[:, 1::2] = torch.cos(position * div_term) # 홀수 인덱스
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.query_linear = nn.Linear(d_model, d_model)
        self.key_linear = nn.Linear(d_model, d_model)
        self.value_linear = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)

    def attention(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        out = torch.matmul(p_attn, value)
        return out

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query = self.query_linear(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = self.key_linear(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = self.value_linear(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output = self.attention(query, key, value, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.output_linear(attn_output)
    

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.input_layer = nn.Linear(d_model, d_ff)
        self.output_layer = nn.Linear(d_ff, d_model)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x



class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        # src_vocab_size: Source vocabulary size
        # tgt_vocab_size: Target vocabulary size
        # d_model: Dimensionality of the embeddings
        # num_heads: Number of attention heads
        # d_ff: Dimensionality of the feed-forward layer
        # num_layers: Number of encoder/decoder layers
        # dropout: Dropout rate
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout)
        self.generator = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask, tgt_mask):
        # src_mask : decoder cross-attn에서 특정 위치 마스킹 목적 (패딩 토큰 등)
        # tgt_mask : decoder self-attn에서 미래 시점 단어를 못보게 하기 위해서
        # memory_mask : Encoder의 출력(memory)에 대해 특정 위치를 마스킹하기 위해
        memory = self.encoder(src, src_mask)

        memory_mask = (src != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)

        output = self.decoder(tgt, memory, tgt_mask, memory_mask)
        return self.generator(output)
    

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len = max_length)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, src, mask):
        x = self.embedding(src)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self-attention with residual connection and normalization
        attention_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attention_output))

        # Feed-forward with residual connection and normalization
        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(feed_forward_output))

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=max_length)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])

    def forward(self, tgt, memory, tgt_mask, memory_mask):
        x = self.embedding(tgt)
        x = self.positional_encoding(x)
        # n_layers 수만큼 반복
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, memory_mask)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, memory, tgt_mask, memory_mask):
        x2 = self.norm1(x + self.dropout1(self.self_attn(x, x, x, tgt_mask)))
        x3 = self.norm2(x2 + self.dropout2(self.cross_attn(x2, memory, memory, memory_mask )))
        out = self.norm3(x3 + self.dropout3(self.feed_forward(x3)))
        return out


# Custom Dataset
class TranslationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length, device):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

        # Load data
        try:
            data = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Failed to read data from {file_path}: {e}")

        self.src_texts = data['eng'].tolist()
        self.tgt_texts = data['kor'].tolist()

    def __len__(self):
        return len(self.src_texts)

    def __getitem__(self, idx):
        src = self.tokenizer(self.src_texts[idx],return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)['input_ids'].squeeze(0).to(self.device)
        tgt = self.tokenizer(self.tgt_texts[idx],return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)['input_ids'].squeeze(0).to(self.device)
        return src, tgt

# Subsequent Mask
def subsequent_mask(size):
    # size: Length of the sequence
    mask = torch.triu(torch.ones(size, size), diagonal=1).type(torch.bool)
    return mask == 0

# Collate function for DataLoader
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_batch = torch.stack(src_batch)
    tgt_batch = torch.stack(tgt_batch)
    return src_batch, tgt_batch

# Main function
if __name__ == "__main__":
    src_vocab_size = 30522  # BERT tokenizer vocab size
    tgt_vocab_size = 30522  # BERT tokenizer vocab size
    d_model = 512  # Dimensionality of the embeddings
    num_heads = 8  # Number of attention heads
    d_ff = 2048  # Dimensionality of the feed-forward layer
    num_layers = 6  # Number of encoder/decoder layers
    dropout = 0.1  # Dropout rate
    max_length = 50  # Maximum length of tokenized sequences
    batch_size = 32  # Batch size
    num_epochs = 10  # Number of epochs

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset and dataloader
    dataset = TranslationDataset("./data/bible_data.csv", tokenizer, max_length, device)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Initialize Transformer model
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(num_epochs):
        transformer.train()
        epoch_loss = 0
        total_tokens = 0

        for batch_idx, (src, tgt) in enumerate(dataloader):
            optimizer.zero_grad()

             # src_mask 생성
            src_mask = (src != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)

            # tgt_mask 생성
            look_ahead_mask = subsequent_mask(tgt.size(1)).to(device)  # (seq_len, seq_len)
            padding_mask = (tgt != tokenizer.pad_token_id).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
            tgt_mask = look_ahead_mask.unsqueeze(0).unsqueeze(1) & padding_mask  # (batch_size, num_heads, seq_len, seq_len)

           # Separate target into input and output
            tgt_input = tgt[:, :-1]  # Decoder input
            tgt_output = tgt[:, 1:]  # Expected output

            # Model forward pass
            output = transformer(src, tgt_input, src_mask, tgt_mask)

            # Reshape outputs for loss calculation
            output = output.view(-1, tgt_vocab_size)
            tgt_output = tgt_output.contiguous().view(-1)

            # Compute loss
            loss = criterion(output, tgt_output)
            loss.backward()
            optimizer.step()

            # Track loss and token count
            epoch_loss += loss.item() * tgt_output.size(0)
            total_tokens += tgt_output.size(0)

        avg_loss = epoch_loss / total_tokens
        print(f"Epoch {epoch + 1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    print("Training complete.")


