# Roman Urdu Poetry Generator

This project implements a Roman Urdu Poetry Generator using TensorFlow, NLP techniques, and Gradio for an interactive interface. The model learns from a dataset of Roman Urdu poetry and generates new verses while preserving poetic structure.

##  Project Overview

- **Dataset**: CSV file with poetry in Roman Urdu.
- **Model**: LSTM-based language model.
- **Interface**: Gradio for user interaction.

##  Directory Structure

```
.
├── dataset.csv            # Poetry dataset
├── Model-Training-AI.ipynb     # Model training and inference
├── app.py                  # Gradio interface
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

##  Installation

1. **Clone the Repository**

```bash
git clone https://github.com/your-username/roman-urdu-poetry-generator.git
cd roman-urdu-poetry-generator
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

##  Usage

1. **Run the Web Interface**

```bash
python app.py
```

2. **Access the Interface**

- Open your browser and go to `http://localhost:7860/`

##  Model Architecture

- **Embedding Layer**: Word embeddings
- **LSTM Layer**: Sequential pattern learning
- **Dense Layer**: Word predictions

##  Gradio Interface

- Input: Seed word or phrase
- Parameters: Number of words, Temperature
- Output: Generated poetry

##  Example Output

**Input**: "dil"

**Output**:

```
dil ne kuch aise jazbaat paaye
ishq mein hum ne ansoo bahaaye
```

##  Customization

- Adjust `EPOCHS`, `BATCH_SIZE` in `poetry_generator.py`.
- Modify dataset by updating `dataset.csv`.

##  Troubleshooting

- Ensure dataset has a `Poetry` column.
- Adjust max verse length if generation quality is poor.



**Happy Poetry Generation! 🎤**

