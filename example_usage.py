#!/usr/bin/env python3
"""
Example usage of Transformer Password Generator
This is a demonstration script showing the concept and architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
import random

class CharTokenizer:
    """Simple character-level tokenizer for demonstration"""
    
    def __init__(self, char_set='ascii_printable'):
        """Initialize tokenizer with ASCII printable characters"""
        if char_set == 'ascii_printable':
            # 95 printable ASCII characters (32-126)
            self.chars = [chr(i) for i in range(32, 127)]
        
        self.char_to_idx = {char: idx for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.pad_token = 0
        
        print(f"Tokenizer initialized with {self.vocab_size} characters")
    
    def encode(self, text: str, max_length: int = 16) -> List[int]:
        """Encode text to indices with padding/truncation"""
        indices = [self.char_to_idx.get(char, 1) for char in text]  # 1 is UNK token
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices += [self.pad_token] * (max_length - len(indices))
        return indices
    
    def decode(self, indices: List[int]) -> str:
        """Decode indices back to text"""
        return ''.join([self.idx_to_char.get(idx, '?') for idx in indices if idx != self.pad_token])

class TransformerModel(nn.Module):
    """Simplified Transformer model for demonstration"""
    
    def __init__(self, vocab_size: int, d_model: int = 256, nhead: int = 4, 
                 num_layers: int = 4, dim_feedforward: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1000, d_model))
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        batch_size, seq_len = x.shape
        
        # Embedding
        x = self.embedding(x) * (self.d_model ** 0.5)
        
        # Position encoding
        if seq_len <= self.pos_encoding.size(1):
            pos_enc = self.pos_encoding[:, :seq_len, :]
        else:
            pos_enc = self.pos_encoding[:, :seq_len, :]
        x = x + pos_enc
        
        # Create causal mask for autoregressive generation
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = mask.to(x.device)
        
        # Transformer decoder
        x = self.transformer(x, x, tgt_mask=mask)
        x = self.dropout(x)
        
        # Output projection
        logits = self.output_projection(x)
        return logits

def generate_passwords_demo(model, tokenizer, num_passwords=5, max_length=12,
                          temperature=1.0, top_k=50, device='cpu'):
    """Generate passwords using the trained model (demonstration)"""
    model.eval()
    passwords = []
    
    print(f"\n🔐 Generating {num_passwords} passwords (demo mode):")
    print("=" * 50)
    
    with torch.no_grad():
        for i in range(num_passwords):
            # Start with a random character
            current_seq = torch.tensor([[random.randint(0, tokenizer.vocab_size-1)]], 
                                     dtype=torch.long, device=device)
            
            generated_chars = []
            for _ in range(max_length - 1):
                # Forward pass
                logits = model(current_seq)
                
                # Get logits for the last token
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Sample from the distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append to sequence
                current_seq = torch.cat([current_seq, next_token.unsqueeze(0)], dim=1)
                generated_chars.append(tokenizer.idx_to_char.get(next_token.item(), '?'))
                
                # Stop if we hit the end token
                if next_token.item() == tokenizer.pad_token:
                    break
            
            # Create the password
            password = ''.join(generated_chars)
            passwords.append(password)
            print(f"{i+1:2d}. {password}")
    
    return passwords

def main():
    """Main demonstration function"""
    print("🔐 Transformer Password Generator - Demo")
    print("=" * 50)
    print("This is a demonstration of the concept and architecture.")
    print("The actual training code is not included in this repository.")
    print()
    
    # Initialize tokenizer
    print("📝 Initializing tokenizer...")
    tokenizer = CharTokenizer()
    
    # Initialize model (untrained for demo)
    print("🤖 Initializing model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerModel(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        nhead=4,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1
    ).to(device)
    
    print(f"✅ Model initialized on {device}")
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Demonstrate password generation (with untrained model)
    print("⚠️  Note: This model is untrained, so generated passwords are random.")
    print("   In a real implementation, the model would be trained on password datasets.")
    print()
    
    passwords = generate_passwords_demo(
        model=model,
        tokenizer=tokenizer,
        num_passwords=10,
        max_length=12,
        temperature=1.0,
        top_k=50,
        device=device
    )
    
    print("\n" + "=" * 50)
    print("🎯 Key Features Demonstrated:")
    print("• Character-level tokenization")
    print("• Transformer architecture")
    print("• Autoregressive generation")
    print("• Temperature and top-k sampling")
    print()
    print("📚 Educational Value:")
    print("• Understanding password patterns")
    print("• AI pattern recognition")
    print("• Cybersecurity research")
    print("• Character-level language modeling")
    print()
    print("⚠️  Remember: This tool is for educational purposes only!")
    print("   Always use responsibly and in compliance with applicable laws.")

if __name__ == "__main__":
    main()
