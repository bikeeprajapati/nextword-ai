# 🚀 NextWord AI - Transfer Learning Comparison

> Two implementations of next-word prediction: From fundamentals to state-of-the-art

**A learning journey through PyTorch, Transfer Learning, and Modern NLP**

---

## 🎯 Project Overview

This project implements next-word prediction in TWO ways:

### **Version 1: LSTM + GloVe Embeddings**
- Built from scratch using PyTorch
- Pre-trained GloVe word embeddings
- LSTM neural network
- **Goal:** Understand fundamentals

### **Version 2: Fine-tuned GPT-2**
- Hugging Face Transformers
- Pre-trained GPT-2 model
- Fine-tuned on custom data
- **Goal:** Production-quality results

### **Why Both?**
- **Learn** how neural networks work (Version 1)
- **Use** state-of-the-art methods (Version 2)
- **Compare** approaches (educational value)
- **Show** progression in portfolio

---

## 📊 Results Comparison

| Metric | Version 1 (GloVe) | Version 2 (GPT-2) |
|--------|-------------------|-------------------|
| **Accuracy** | ~68% | ~85% |
| **Training Time** | 2-3 hours (CPU) | 1-2 hours (CPU) |
| **Model Size** | 15 MB | 500 MB |
| **Inference Speed** | Fast | Medium |
| **Cold Start** | Needs training | Pre-trained |
| **Customization** | Full control | Limited |
| **Learning Value** | High | Medium |

---

## 🏗️ Architecture

### Version 1: Custom LSTM
```
Input Text
    ↓
Tokenization (word → ID)
    ↓
GloVe Embedding (ID → 100D vector)
    ↓
LSTM (2 layers, 256 hidden)
    ↓
Dense Layer
    ↓
Softmax (probability distribution)
    ↓
Next Word Prediction
```

### Version 2: GPT-2
```
Input Text
    ↓
GPT-2 Tokenizer
    ↓
Pre-trained GPT-2 (117M parameters)
    ↓
Fine-tuning on custom data
    ↓
Next Word Prediction
```

---

## 📦 Installation

### Prerequisites
- Python 3.9+
- 8GB RAM minimum
- 2GB free disk space

### Setup

```bash
# 1. Clone repository
git clone https://github.com/bikeeprajapati/nextword-ai.git
cd nextword-ai

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Prepare data
python prepare_data.py
```

---

## 🚀 Quick Start

### Run Both Versions

```bash
# Version 1: LSTM + GloVe
python version1_glove/train.py

# Version 2: GPT-2
python version2_gpt2/fine_tune.py

# Compare in Streamlit
streamlit run app.py
```

### Try Predictions

```python
# Version 1
from version1_glove.predict import predict_next_word

text = "I love machine"
prediction = predict_next_word(text)
print(prediction)  # "learning" (68% confidence)

# Version 2
from version2_gpt2.predict import predict_gpt2

prediction = predict_gpt2(text)
print(prediction)  # "learning" (85% confidence)
```

---

## 📚 Learning Path

### Week 1: PyTorch Fundamentals
- [x] Day 1: Tensors & Operations
- [x] Day 2: Neural Networks
- [x] Day 3: Text Processing
- [x] Day 4: LSTM Architecture
- [x] Day 5: Training Loop

### Week 2: Version 1 (GloVe)
- [x] Download & process GloVe
- [x] Build LSTM model
- [x] Training pipeline
- [x] Evaluation
- [x] Predictions

### Week 3: Version 2 (GPT-2)
- [x] Hugging Face setup
- [x] Data preparation
- [x] Fine-tuning
- [x] Evaluation
- [x] Deployment

### Week 4: Comparison & Deploy
- [x] Side-by-side comparison
- [x] Streamlit interface
- [x] Documentation
- [x] Demo video

---

## 🎓 Key Learnings

### Technical Skills
✅ PyTorch fundamentals (tensors, autograd, nn.Module)
✅ LSTM/RNN architectures
✅ Word embeddings (GloVe)
✅ Transfer learning concepts
✅ Transformer models (GPT-2)
✅ Fine-tuning pre-trained models
✅ Model evaluation & comparison
✅ Streamlit deployment

### Concepts Mastered
- Sequence modeling
- Language modeling
- Tokenization strategies
- Embedding spaces
- Attention mechanisms (GPT-2)
- Overfitting prevention
- Hyperparameter tuning

---

## 📁 Project Structure

```
nextword-ai/
├── lessons/                          # Learning materials
│   ├── lesson1_tensors.py
│   ├── lesson2_neural_network.py
│   ├── transfer_learning.py
│   └── advanced_transfer.py
│
├── version1_glove/                   # LSTM + GloVe
│   ├── dataset.py                    # Data loading
│   ├── model.py                      # LSTM model
│   ├── train.py                      # Training script
│   └── predict.py                    # Inference
│
├── version2_gpt2/                    # GPT-2
│   ├── fine_tune.py                  # Fine-tuning script
│   └── predict.py                    # Inference
│
├── data/
│   ├── sample_text.txt               # Training data
│   ├── sequences.txt                 # Processed sequences
│   ├── vocabulary.json               # Word mappings
│   └── glove/                        # GloVe embeddings
│
├── models/
│   ├── glove_model.pth               # Trained LSTM
│   └── gpt2_finetuned/               # Fine-tuned GPT-2
│
├── prepare_data.py                   # Data preparation
├── app.py                            # Streamlit app
├── requirements.txt
└── README.md
```

---

## 🔧 Configuration

### Version 1 Hyperparameters
```python
EMBEDDING_DIM = 100      # GloVe dimension
HIDDEN_DIM = 256         # LSTM hidden size
NUM_LAYERS = 2           # LSTM layers
DROPOUT = 0.3            # Dropout rate
LEARNING_RATE = 0.001    # Adam LR
BATCH_SIZE = 32
EPOCHS = 10
```

### Version 2 Hyperparameters
```python
MODEL_NAME = 'gpt2'      # or 'gpt2-medium'
MAX_LENGTH = 128         # Sequence length
LEARNING_RATE = 5e-5     # Fine-tuning LR
BATCH_SIZE = 4           # Smaller for GPT-2
EPOCHS = 3               # Usually 2-5 is enough
```

---

## 📊 Performance Analysis

### Version 1 (GloVe + LSTM)

**Strengths:**
- ✅ Fast inference
- ✅ Small model size
- ✅ Full control over architecture
- ✅ Interpretable
- ✅ Good learning experience

**Weaknesses:**
- ❌ Lower accuracy
- ❌ Limited context window
- ❌ Needs more training data
- ❌ Doesn't handle rare words well

### Version 2 (GPT-2)

**Strengths:**
- ✅ High accuracy
- ✅ Better context understanding
- ✅ Handles rare words
- ✅ Production-ready
- ✅ Easy to implement

**Weaknesses:**
- ❌ Larger model size
- ❌ Slower inference
- ❌ Less interpretable
- ❌ Requires more resources

---

## 🎯 Use Cases

### Version 1 is Better For:
- Educational purposes
- Resource-constrained environments
- Real-time applications
- Custom architectures
- Understanding fundamentals

### Version 2 is Better For:
- Production systems
- Best accuracy needed
- Complex language tasks
- When computational resources available
- Quick deployment

---

## 🚧 Future Enhancements

### Short Term
- [ ] Add beam search
- [ ] Temperature sampling
- [ ] Top-k/top-p sampling
- [ ] Comparison metrics dashboard
- [ ] Model interpretability tools

### Long Term
- [ ] Try other embeddings (FastText, BERT)
- [ ] Implement attention visualization
- [ ] Add more pre-trained models (GPT-Neo, OPT)
- [ ] Create REST API
- [ ] Mobile app
- [ ] Multi-language support

---

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Improve data preprocessing
- Add more pre-trained models
- Enhance Streamlit UI
- Add evaluation metrics
- Write tutorials

---

## 📝 License

MIT License - See LICENSE file

---

## 👨‍💻 Author

**Bikee Prajapati**
- GitHub: [@bikeeprajapati](https://github.com/bikeeprajapati)
- LinkedIn: [Bikee Prajapati](https://linkedin.com/in/bikeeprajapati)
- Email: bikeeprajapati1@gmail.com
- Website: [bikeeprajapati.com.np](https://bikeeprajapati.com.np)

**Institution:** Shanker Dev Campus, Kathmandu
**Program:** Bachelor's in Information Management
**Focus:** AI/ML, NLP, Deep Learning

---

## 🙏 Acknowledgments

- Stanford NLP Group (GloVe embeddings)
- Hugging Face (Transformers library)
- PyTorch Team
- OpenAI (GPT-2)
- Project Gutenberg (training data)

---

## 📖 References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformers
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2

---

## 📊 Project Stats

- **Lines of Code:** ~2,500+
- **Training Time:** 3-5 hours total
- **Models:** 2 (LSTM + GPT-2)
- **Approaches:** From scratch + Transfer learning
- **Learning Value:** ⭐⭐⭐⭐⭐

---

⭐ **Star this repo if you're learning PyTorch and NLP!**

**Built with ❤️ in Kathmandu | Learning by Building | AI/ML Student Portfolio**