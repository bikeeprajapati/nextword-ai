"""
NextWord AI - Data Preparation
Prepare training data for both GloVe and GPT-2 versions

This script:
1. Downloads sample text data
2. Cleans and preprocesses
3. Creates training sequences
4. Saves for both versions
"""

import os
import re
import requests
from typing import List, Tuple
import json

print("=" * 70)
print("📊 DATA PREPARATION FOR NEXTWORD AI")
print("=" * 70)

# ============================================================================
# PART 1: Download Sample Data
# ============================================================================

def download_sample_data(data_dir='data'):
    """
    Download sample text for training
    Using Shakespeare's works (public domain, good for learning)
    """
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, 'sample_text.txt')
    
    if os.path.exists(file_path):
        print("✅ Sample data already exists!")
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        return text
    
    print("📥 Downloading Shakespeare's works...")
    
    # Shakespeare from Project Gutenberg
    url = "https://www.gutenberg.org/files/100/100-0.txt"
    
    try:
        response = requests.get(url)
        text = response.text
        
        # Save to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"✅ Downloaded and saved to {file_path}")
        print(f"   Size: {len(text):,} characters")
        
        return text
    
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("💡 Creating sample data manually...")
        
        # Fallback: Create sample text
        sample_text = """
        Machine learning is a subset of artificial intelligence.
        Deep learning uses neural networks with many layers.
        Python is a popular programming language for data science.
        Natural language processing enables computers to understand text.
        Transfer learning allows us to use pre-trained models.
        PyTorch is a powerful framework for building neural networks.
        The next word prediction task is fundamental to language modeling.
        Recurrent neural networks can process sequential data effectively.
        LSTM networks solve the vanishing gradient problem.
        Embeddings represent words as dense vectors in continuous space.
        """ * 100  # Repeat to have more data
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        
        print(f"✅ Created sample data: {file_path}")
        return sample_text


# ============================================================================
# PART 2: Text Cleaning
# ============================================================================

def clean_text(text: str) -> str:
    """
    Clean and preprocess text
    
    Steps:
    1. Remove special characters (keep periods, commas)
    2. Convert to lowercase
    3. Remove extra whitespace
    4. Remove numbers (optional)
    """
    print("\n🧹 Cleaning text...")
    
    original_length = len(text)
    
    # Remove Gutenberg header/footer
    start_marker = "*** START OF"
    end_marker = "*** END OF"
    if start_marker in text and end_marker in text:
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        text = text[start_idx:end_idx]
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\;\:\-]', '', text)
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove multiple periods
    text = re.sub(r'\.+', '.', text)
    
    text = text.strip()
    
    print(f"   Original: {original_length:,} characters")
    print(f"   Cleaned:  {len(text):,} characters")
    print(f"   Preview: {text[:200]}...")
    
    return text


# ============================================================================
# PART 3: Create Training Sequences
# ============================================================================

def create_sequences(text: str, sequence_length: int = 5) -> List[Tuple[str, str]]:
    """
    Create training sequences for next-word prediction
    
    Example:
        Input text: "I love machine learning and AI"
        Sequence length: 3
        
        Sequences:
        ("I love machine", "learning")
        ("love machine learning", "and")
        ("machine learning and", "AI")
    
    Args:
        text: Cleaned text
        sequence_length: Number of words in input sequence
    
    Returns:
        List of (input_sequence, target_word) tuples
    """
    print(f"\n📝 Creating sequences (length={sequence_length})...")
    
    # Split into words
    words = text.split()
    
    sequences = []
    
    # Create sliding window
    for i in range(len(words) - sequence_length):
        # Input: sequence_length words
        input_seq = ' '.join(words[i:i+sequence_length])
        
        # Target: next word
        target = words[i+sequence_length]
        
        sequences.append((input_seq, target))
    
    print(f"✅ Created {len(sequences):,} training sequences")
    print(f"\nExample sequences:")
    for i, (inp, target) in enumerate(sequences[:5], 1):
        print(f"   {i}. Input: '{inp}' → Target: '{target}'")
    
    return sequences


# ============================================================================
# PART 4: Build Vocabulary
# ============================================================================

def build_vocabulary(text: str, min_frequency: int = 2) -> dict:
    """
    Build word-to-index vocabulary
    
    Args:
        text: Cleaned text
        min_frequency: Minimum word frequency to include
    
    Returns:
        word_to_idx: Dictionary mapping words to indices
    """
    print(f"\n📚 Building vocabulary (min_frequency={min_frequency})...")
    
    # Count word frequencies
    from collections import Counter
    words = text.split()
    word_counts = Counter(words)
    
    # Filter by minimum frequency
    vocab_words = [word for word, count in word_counts.items() 
                if count >= min_frequency]
    
    # Sort for consistency
    vocab_words = sorted(vocab_words)
    
    # Create word-to-index mapping
    word_to_idx = {
        '<PAD>': 0,      # Padding token
        '<UNK>': 1,      # Unknown token
    }
    
    for idx, word in enumerate(vocab_words, start=2):
        word_to_idx[word] = idx
    
    # Create reverse mapping
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    print(f"✅ Vocabulary size: {len(word_to_idx):,} words")
    print(f"   Total unique words: {len(word_counts):,}")
    print(f"   Filtered out: {len(word_counts) - len(vocab_words):,} rare words")
    
    print(f"\nMost common words:")
    for word, count in word_counts.most_common(10):
        print(f"   '{word}': {count:,} times")
    
    return word_to_idx, idx_to_word


# ============================================================================
# PART 5: Save Processed Data
# ============================================================================

def save_processed_data(sequences, word_to_idx, idx_to_word, data_dir='data'):
    """
    Save processed data for training
    """
    print(f"\n💾 Saving processed data...")
    
    os.makedirs(data_dir, exist_ok=True)
    
    # Save sequences
    sequences_file = os.path.join(data_dir, 'sequences.txt')
    with open(sequences_file, 'w', encoding='utf-8') as f:
        for input_seq, target in sequences:
            f.write(f"{input_seq}\t{target}\n")
    
    print(f"✅ Saved sequences: {sequences_file}")
    
    # Save vocabulary
    vocab_file = os.path.join(data_dir, 'vocabulary.json')
    with open(vocab_file, 'w', encoding='utf-8') as f:
        json.dump({
            'word_to_idx': word_to_idx,
            'idx_to_word': idx_to_word,
            'vocab_size': len(word_to_idx)
        }, f, indent=2)
    
    print(f"✅ Saved vocabulary: {vocab_file}")
    
    # Save statistics
    stats = {
        'total_sequences': len(sequences),
        'vocab_size': len(word_to_idx),
        'sequence_length': len(sequences[0][0].split()) if sequences else 0
    }
    
    stats_file = os.path.join(data_dir, 'stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Saved statistics: {stats_file}")
    
    return sequences_file, vocab_file


# ============================================================================
# PART 6: Data Statistics
# ============================================================================

def print_data_statistics(text, sequences, word_to_idx):
    """
    Print comprehensive data statistics
    """
    print("\n" + "=" * 70)
    print("📊 DATA STATISTICS")
    print("=" * 70)
    
    words = text.split()
    unique_words = len(set(words))
    
    print(f"\nRaw Data:")
    print(f"  • Total characters: {len(text):,}")
    print(f"  • Total words: {len(words):,}")
    print(f"  • Unique words: {unique_words:,}")
    
    if len(words) > 0:
        print(f"  • Average word length: {sum(len(w) for w in words) / len(words):.1f} chars")
    else:
        print(f"  • Average word length: N/A (no words found)")
    
    print(f"\nProcessed Data:")
    print(f"  • Training sequences: {len(sequences):,}")
    print(f"  • Vocabulary size: {len(word_to_idx):,}")
    print(f"  • Coverage: {len(word_to_idx) / unique_words * 100:.1f}%")
    
    # Estimate training time
    epochs = 10
    batch_size = 32
    batches_per_epoch = len(sequences) // batch_size
    total_batches = batches_per_epoch * epochs
    
    print(f"\nTraining Estimates (10 epochs, batch_size=32):")
    print(f"  • Batches per epoch: {batches_per_epoch:,}")
    print(f"  • Total batches: {total_batches:,}")
    print(f"  • Estimated time (CPU): ~{total_batches * 0.1 / 60:.0f} minutes")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main data preparation pipeline
    """
    print("\n🚀 Starting data preparation pipeline...\n")
    
    # Step 1: Download data
    print("STEP 1: Download Data")
    print("-" * 70)
    text = download_sample_data()
    
    # Step 2: Clean text
    print("\nSTEP 2: Clean Text")
    print("-" * 70)
    cleaned_text = clean_text(text)
    
    # Step 3: Create sequences
    print("\nSTEP 3: Create Sequences")
    print("-" * 70)
    sequences = create_sequences(cleaned_text, sequence_length=5)
    
    # Step 4: Build vocabulary
    print("\nSTEP 4: Build Vocabulary")
    print("-" * 70)
    word_to_idx, idx_to_word = build_vocabulary(cleaned_text, min_frequency=2)
    
    # Step 5: Save data
    print("\nSTEP 5: Save Processed Data")
    print("-" * 70)
    save_processed_data(sequences, word_to_idx, idx_to_word)
    
    # Step 6: Statistics
    print_data_statistics(cleaned_text, sequences, word_to_idx)
    
    print("\n" + "=" * 70)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 70)
    
    print("""
    Next Steps:
    1. Version 1: Train LSTM with GloVe embeddings
    → python version1_glove/train.py
    
    2. Version 2: Fine-tune GPT-2
    → python version2_gpt2/fine_tune.py
    
    3. Compare both versions in Streamlit
    → streamlit run app.py
    """)


if __name__ == "__main__":
    main()