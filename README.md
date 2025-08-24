# Pokemon Creator - WORK IN PROGRESS

A machine learning-powered Pokemon creator that predicts stats and generates Pokemon. Used and created a data pipeline from the data provided by [PokéAPI](https://pokeapi.co/).

## Project Status

### Working Features
- **ML Stat Prediction**: Random Forest models predict missing Pokemon stats with 95% confidence intervals
- **Interactive Web App**: Built with Streamlit for easy Pokemon creation
- **Dataset Explorer**: Browse and analyze 1000+ Pokemon dataset
- **Model Analytics**: Detailed Random Forest performance metrics
- **Enhanced Validation System**: Real-time warnings for unrealistic Pokemon stats

### Under Development
- **AI Image Generation**: Stable Diffusion integration (code complete, available for local development)
- **AI Text Generation**: DistilGPT-2 name generation (code complete, available for local development)

## Technical Implementation

### Machine Learning Pipeline
- **Algorithm**: Random Forest Regressor (100 trees per feature)
- **Features**: Height, weight, base experience, HP, attack, defense, special attack, special defense, speed
- **Confidence Intervals**: 95% prediction intervals using tree ensemble variance
- **Missing Value Handling**: Median imputation strategy
- Models get updated as user inputs their own stats

### Preserved AI Features (Local Development)
The codebase includes complete implementations for:
- Custom Pokemon image generation with stat-based prompt engineering
- AI text generation with procedural fallbacks
- Memory optimization for large model inference
- Progressive loading with user feedback

## Usage

### Local Development (Full Features)
```bash
# Clone repository
git clone https://github.com/NoahDPJ03/pokemon-creator.git
cd pokemon-creator

# Install dependencies for full AI features
pip install -r requirements.txt
pip install torch diffusers transformers

# Enable AI features in Modeling.py:
# 1. Uncomment AI import statements
# 2. Set TORCH_AVAILABLE = True
# 3. Uncomment AI generation functions

# Run locally
streamlit run Modeling.py
```

## Learning Outcomes

This project demonstrates:
- **ML Engineering**: Building production-ready prediction models with confidence intervals
- **AI Integration**: Implementing Stable Diffusion and language models (preserved in code)
- **Web Development**: Creating interactive data science applications with Streamlit
- **Deployment Strategy**: Adapting complex AI applications for cloud hosting constraints
- **Code Preservation**: Maintaining complete feature implementations for portfolio purposes

### What I Learned About Deployment

- Free hosting services have strict memory limits that prevent large AI model usage
- Streamlit Community Cloud doesn't support GPU acceleration
- It's important to have fallback strategies for resource-intensive features
- Code preservation is valuable even when features must be disabled

## Dataset

The Pokemon dataset contains comprehensive information about Pokemon including:
- Physical characteristics (height, weight)
- Base stats (HP, Attack, Defense, Special Attack, Special Defense, Speed)
- Experience values and other metadata

Data sourced from [PokéAPI](https://pokeapi.co/)

## How It Works

1. **Input**: Enter known Pokemon statistics in the input fields
2. **Prediction**: Random Forest models predict missing values with confidence intervals
3. **Visualization**: See predictions with uncertainty bounds
4. **Analysis**: Explore model performance and feature importance

The ML models use a pipeline with median imputation for missing values and Random Forest regression with 100 trees per feature.

## AI Models (Under Development)

- **Name Generation**: DistilGPT-2 with procedural fallback system - In Progress
- **Image Generation**: Stable Diffusion v1.5 with detailed stat-based prompting - In Progress
- **Stat Prediction**: Random Forest ensemble with 100 trees per feature - Working

## Model Performance

- **Algorithm**: Random Forest Regressor
- **Trees per model**: 100 
- **Cross-validation**: Built-in through ensemble averaging
- **Confidence intervals**: Calculated from tree prediction variance
- **Features**: 9 Pokemon statistics

## Development Status

- **ML Prediction Engine**: Fully functional
- **Web Interface**: Complete and responsive  
- **Data Pipeline**: Working with cleaned Pokemon dataset
- **AI Image Generation**: Under development
- **AI Name Generation**: Under development
- **Deployment**: Optimizing for cloud hosting

## Project Structure

```
pokemon-creator/
├── Modeling.py              # Main Streamlit app
├── Cleaning.ipynb          # Data preprocessing notebook
├── Modeling.ipynb          # Model development notebook
├── pokemon_data.csv        # Raw Pokemon data
├── pokemon_data_cleaned.csv # Processed data
└── requirements.txt        # Dependencies
```

## Development

The project includes Jupyter notebooks for data exploration and model development:
- `Cleaning.ipynb`: Data preprocessing and feature engineering
- `Modeling.ipynb`: Model training and evaluation

## Contributing

This is a work-in-progress project. Contributions welcome!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source and available under the [MIT License](LICENSE).

## Deployment

### Current Status
The ML prediction functionality works reliably and can be deployed to:
- Streamlit Community Cloud
- Heroku
- Railway
- Any Python hosting service

AI features are still being developed and tested.

## Notes

- The prediction engine is production-ready
- AI image/name generation features are experimental
- Local development environment required for full feature testing

## Acknowledgments

- Pokemon data from [PokéAPI](https://pokeapi.co/)
- Image generation powered by [Hugging Face Diffusers](https://github.com/huggingface/diffusers) - In Development
- Built with [Streamlit](https://streamlit.io/)
