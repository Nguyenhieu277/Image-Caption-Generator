# Image Caption Generator

This is an image caption generator using a pre-trained DenseNet201 model for feature extraction and a deep learning model for caption generation. The project is built with Streamlit for a simple and interactive web application.

## 🚀 Features
- Upload an image and generate a caption automatically.
- Uses DenseNet201 for feature extraction.
- Implements beam search for better caption generation.
- Streamlit-based web interface for easy interaction.
- Flickr8k dataset `https://www.kaggle.com/datasets/adityajn105/flickr8k`
## 📦 Installation

### Prerequisites
Ensure you have Python installed (recommended version: 3.8+).

### Clone the repository
```bash
git clone https://github.com/your-repo/image-caption-generator.git
cd image-caption-generator
```

### Install dependencies
```bash
pip install -r requirements.txt
```

## 🔧 Usage

### Run the application
```bash
streamlit run src/main.py
```

### Upload an image
1. Click on the "Upload an Image" button.
2. Select a `.jpg`, `.png`, or `.jpeg` file.
3. The model will process the image and display the generated caption.

## 🏗 Project Structure
```
image-caption-generator/
|dataset
│── src/
│   ├── main.py            # Main application script
├── trained_model/         # Folder containing trained models & tokenizer
│── requirements.txt       # Required dependencies
│── README.md              # Project documentation
```

## 📖 Model Details
- **Feature Extraction:** Uses DenseNet201 to extract image features.
- **Caption Generation:** Uses a trained LSTM model to generate captions.
- **Beam Search:** Improves the caption quality by considering multiple possible sequences.

## ⚡ Common Issues & Fixes
### 1. `No module named streamlit`
Run:
```bash
pip install streamlit
```

### 2. `Error loading model: File not found`
Ensure that the trained models (`tokenizer.pkl`, `caption_imgs.keras`, `features.pkl`) are available in `trained_model/`.



