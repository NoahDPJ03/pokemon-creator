# 🔥 Pokemon Creator - AI-Powered Pokemon Generator

Create your own custom Pokemon with ML-predicted stats, AI-generated images, and intelligent names!

## 🌟 Features

- **📊 Smart Prediction**: ML models predict missing Pokemon stats with 95% confidence intervals
- **🎨 AI Image Generation**: Stable Diffusion creates unique Pokemon artwork based on your stats
- **🏷️ Intelligent Naming**: Hybrid AI + procedural system generates Pokemon-appropriate names
- **📈 Data Analytics**: Explore the complete Pokemon dataset and model insights

## 🚀 Try It Live

Visit the live app: [Coming Soon - Deploy to Streamlit Cloud]

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **ML Models**: Random Forest (scikit-learn)
- **Image Generation**: Stable Diffusion v1.5
- **Text Generation**: DistilGPT-2
- **Data Processing**: Pandas, NumPy

## 📋 Local Setup

1. Clone the repository:
```bash
git clone https://github.com/NoahDPJ03/pokemon-creator.git
cd pokemon-creator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run Modeling.py
```

## 📊 Dataset

Pokemon data sourced from [PokéAPI](https://pokeapi.co/) with 9 statistical features including:
- Height, Weight, Base Experience
- HP, Attack, Defense, Special Attack, Special Defense, Speed

## 🎯 How It Works

1. **Input Pokemon Stats**: Enter known values for any combination of the 9 features
2. **ML Prediction**: Random Forest models predict missing stats with confidence intervals
3. **Generate Content**: Choose to generate names, images, or complete Pokemon profiles
4. **Explore Results**: View detailed generation prompts and model analytics

## 🤖 AI Models

- **Name Generation**: DistilGPT-2 with procedural fallback system
- **Image Generation**: Stable Diffusion v1.5 with detailed stat-based prompting
- **Stat Prediction**: Random Forest ensemble with 100 trees per feature

## 📈 Model Performance

- **Confidence Intervals**: 95% prediction intervals using tree variance
- **Feature Engineering**: Automatic handling of missing values with median imputation
- **Cross-Validation**: Robust model training on complete Pokemon dataset

## 🎨 Example Generations

The app creates Pokemon that reflect their statistical profiles:
- **High Attack**: Aggressive designs with sharp features
- **High Defense**: Armored, fortress-like appearances  
- **High Speed**: Aerodynamic, swift-looking creatures
- **High Special Attack**: Mystical, energy-radiating designs

## 📁 Project Structure

```
pokemon-creator/
├── Modeling.py              # Main Streamlit app
├── Cleaning.ipynb          # Data preprocessing notebook
├── Modeling.ipynb          # Model development notebook
├── pokemon_data.csv        # Raw Pokemon data
├── pokemon_data_cleaned.csv # Processed data
└── requirements.txt        # Dependencies
```

## 🔧 Development

The project includes Jupyter notebooks for data exploration and model development:
- `Cleaning.ipynb`: Data preprocessing and feature engineering
- `Modeling.ipynb`: Model training and evaluation

## 🙏 Acknowledgments

- Pokemon data from [PokéAPI](https://pokeapi.co/)
- Image generation powered by [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- Built with [Streamlit](https://streamlit.io/)
