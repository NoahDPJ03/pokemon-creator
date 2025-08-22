import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import streamlit as st
import matplotlib.pyplot as plt
from diffusers import DiffusionPipeline
from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer
import torch
import re, numpy as np, torch


df = pd.read_csv('pokemon_data_cleaned.csv')


def make_models():
    " Create models for predicting missing features in Pokémon data. "
    features = ['height', 'weight', 'base_experience', 'hp', 'attack', 'defense',
                'special-attack', 'special-defense', 'speed']
    models = {}

    for target in features:
        input_features = [f for f in features if f != target]
        X = df[input_features]
        y = df[target]

        model = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])

        model.fit(X, y)
        models[target] = model
    return models

def predict_missing(user_input, models):
    " Predict missing features based on user input using trained models with confidence intervals. "
    features = ['height', 'weight', 'base_experience', 'hp', 'attack', 'defense',
                'special-attack', 'special-defense', 'speed']
    predictions = {}
    for feature in features:
        if feature in user_input:
            # For user input, use the exact value with no interval
            predictions[feature] = {
                'mean': user_input[feature],
                'lower': user_input[feature],
                'upper': user_input[feature],
                'is_input': True
            }
        else:
            input_features = [f for f in features if f != feature]
            input_vector = [user_input.get(f, None) for f in input_features]
            input_df = pd.DataFrame([input_vector], columns=input_features)

            # Predict the missing feature using the corresponding model
            if None in input_vector:
                input_df = input_df.fillna(df[input_features].median())
            
            # Get individual tree predictions to calculate confidence interval
            model = models[feature]
            
            # Get predictions from all trees in the Random Forest
            tree_predictions = []
            for estimator in model.named_steps['regressor'].estimators_:
                # Transform data through the imputer first
                X_transformed = model.named_steps['imputer'].transform(input_df)
                tree_pred = estimator.predict(X_transformed)[0]
                tree_predictions.append(tree_pred)
            
            # Calculate statistics
            mean_pred = np.mean(tree_predictions)
            std_pred = np.std(tree_predictions)
            
            z_score = 1.96  # for 95% confidence
            margin_error = z_score * std_pred
            
            predictions[feature] = {
                'mean': mean_pred,
                'lower': mean_pred - margin_error,
                'upper': mean_pred + margin_error,
                'is_input': False
            }

    return predictions


_NAME_RE = re.compile(r"^[A-Z][a-z]{3,11}$")

def _llm_name(size_category: str, build_category: str) -> tuple[str | None, str]:
    """Try to get a single clean name from a tiny LM. Returns (name, prompt)."""
    if 'text_generator' not in st.session_state:
        st.session_state.text_generator = pipeline(
            "text-generation",
            model="distilgpt2",
            tokenizer="distilgpt2",
            device=0 if torch.cuda.is_available() else -1
        )

    # Few-shot-ish, with a strict pattern and an obvious stopping cue.
    prompt = (
        "Generate ONE Pokémon-style creature name.\n"
        "Rules: 4-12 letters, alphabetic only, no spaces or punctuation.\n"
        f"Traits: {size_category}, {build_category}.\n"
        "Name: "
    )

    outs = st.session_state.text_generator(
        prompt,
        max_new_tokens=6,              # keep it short
        num_return_sequences=8,        # try a few
        do_sample=True,
        top_p=0.92,
        top_k=50,
        temperature=0.9,
        repetition_penalty=1.1,
        return_full_text=False         # **critical**: don't echo the prompt
    )

    # Extract first word-like token from each completion, filter strictly.
    for o in outs:
        text = o["generated_text"].strip()
        # take the first alphabetic chunk
        m = re.search(r"[A-Za-z]{4,12}", text)
        if not m:
            continue
        candidate = m.group(0).capitalize()
        if _NAME_RE.match(candidate):
            return candidate, prompt
    return None, prompt


def _procedural_name(primary_stat: str) -> str:
    """Deterministic, gamey fallback that always returns a decent name."""
    rng = np.random.default_rng()
    base = {
        'attack':   ["Kr", "Zar", "Dr", "Rav", "Gor", "Blad", "Fang", "Brut"],
        'defense':  ["Gran", "Bulw", "Aeg", "Bast", "For", "Rock", "Stal", "Ward"],
        'speed':    ["Zy", "Velo", "Swift", "Skyr", "Dash", "Gale", "Volt", "Flux"],
        'special-attack': ["My", "Pyro", "Aero", "Lum", "Volt", "Cryo", "Aqua", "Umbr"],
        'hp':       ["Vita", "Endu", "Giga", "Mega", "Stout", "Hearty", "Soma", "Core"]
    }.get(primary_stat, ["Neo", "Nova", "Terra", "Luma", "Astra", "Orbi", "Cosmo"])

    mid = ["ra", "zu", "shi", "gon", "dra", "lix", "zor", "phy", "ri", "ka", "no", "ru"]
    end = ["mon", "chu", "eon", "zor", "dile", "ling", "saur", "rex", "phox", "mite", "loon"]

    parts = [rng.choice(base), rng.choice(mid)]
    if rng.random() < 0.6:  # 2–3 syllables
        parts.append(rng.choice(mid))
    parts.append(rng.choice(end))

    name = "".join(parts).capitalize()
    # trim if somehow longer than 12 letters, retry once
    if len(name) > 12:
        name = (name[:12]).rstrip().capitalize()
    if not _NAME_RE.match(name):
        name = (rng.choice(base) + rng.choice(end)).capitalize()
        name = name[:12]
    return name


def generate_pokemon_name(user_input, progress_callback=None):
    """Generate a Pokemon name based on stats using AI text generation (fixed)."""

    height = user_input.get('height', 0)
    weight = user_input.get('weight', 0)
    hp = user_input.get('hp', 0)
    attack = user_input.get('attack', 0)
    defense = user_input.get('defense', 0)
    special_attack = user_input.get('special-attack', 0)
    special_defense = user_input.get('special-defense', 0)
    speed = user_input.get('speed', 0)

    if progress_callback: progress_callback(10, "Analyzing stats for name generation...")

    primary_stat = max([
        ('attack', attack), ('defense', defense), ('speed', speed),
        ('special-attack', special_attack), ('hp', hp)
    ], key=lambda x: x[1])

    # size/build buckets
    if height > 15:   size_category = "mega"
    elif height > 10: size_category = "titan"
    elif height > 5:  size_category = "normal"
    else:             size_category = "mini"

    ratio = weight / max(height, 1)
    build_category = "heavy" if ratio > 30 else ("balanced" if ratio > 15 else "light")

    if progress_callback: progress_callback(30, "Creating Pokémon name...")

    try:
        if progress_callback: progress_callback(60, "Sampling candidates...")
        name, name_prompt = _llm_name(size_category, build_category)

        if not name:
            # fallback that always returns something decent
            name = _procedural_name(primary_stat[0])
            name_prompt = f"Fallback: Procedural generation using {primary_stat[0]} stat with syllable patterns"

        return {
            'name': name,
            'type_suggestion': get_type_suggestion(user_input),
            'stats_summary': f"Specializes in {primary_stat[0].replace('-', ' ').title()}",
            'name_prompt': name_prompt
        }
    except Exception as e:
        # last-ditch fallback
        return {
            'name': _procedural_name(primary_stat[0]),
            'type_suggestion': 'Normal',
            'stats_summary': 'Balanced fighter',
            'name_prompt': f"Error fallback: Procedural generation using {primary_stat[0]} stat",
            'error': str(e)
        }

def get_type_suggestion(user_input):
    """Suggest Pokemon type based on stats"""
    attack = user_input.get('attack', 0)
    defense = user_input.get('defense', 0)
    special_attack = user_input.get('special-attack', 0)
    special_defense = user_input.get('special-defense', 0)
    speed = user_input.get('speed', 0)
    hp = user_input.get('hp', 0)
    
    # Type suggestions based on stat distributions
    if special_attack > 120:
        if speed > 100:
            return "Electric/Psychic"
        else:
            return "Psychic"
    elif attack > 120:
        if speed > 100:
            return "Fighting/Flying"
        else:
            return "Fighting"
    elif defense > 120:
        if special_defense > 100:
            return "Steel/Rock"
        else:
            return "Steel"
    elif speed > 130:
        return "Electric"
    elif hp > 150:
        return "Normal/Fairy"
    elif special_attack > attack:
        return "Fire/Water"
    else:
        return "Normal"

def generate_pokemon_image_free(user_input, progress_callback=None):
    """Generate a Pokemon image using free Hugging Face Stable Diffusion"""
    
    # Get the final values (user input or predictions)
    height = user_input.get('height', 0)
    weight = user_input.get('weight', 0)
    hp = user_input.get('hp', 0)
    attack = user_input.get('attack', 0)
    defense = user_input.get('defense', 0)
    special_attack = user_input.get('special-attack', 0)
    special_defense = user_input.get('special-defense', 0)
    speed = user_input.get('speed', 0)
    base_experience = user_input.get('base_experience', 0)
    
    if progress_callback:
        progress_callback(10, "Analyzing your Pokémon stats...")

    # Size and build description
    if height > 20:
        size = "massive and towering"
        scale = "giant"
    elif height > 15:
        size = "large and imposing"
        scale = "large"
    elif height > 10:
        size = "medium-sized and well-proportioned"
        scale = "medium"
    elif height > 5:
        size = "compact and agile"
        scale = "small"
    else:
        size = "tiny and adorable"
        scale = "miniature"
    
    # Weight-based build
    weight_height_ratio = weight / max(height, 1)  # Avoid division by zero
    if weight_height_ratio > 50:
        build = "extremely heavy and tank-like"
        body_type = "bulky"
    elif weight_height_ratio > 30:
        build = "heavily built and sturdy"
        body_type = "robust"
    elif weight_height_ratio > 15:
        build = "well-built and balanced"
        body_type = "athletic"
    elif weight_height_ratio > 8:
        build = "lean and agile"
        body_type = "slim"
    else:
        build = "lightweight and graceful"
        body_type = "ethereal"
    
    # Combat style based on attack vs defense
    if attack > defense + 30:
        combat_style = "fierce berserker with razor-sharp claws, fangs, and aggressive stance"
        combat_features = "sharp spikes, claws, aggressive eyes"
    elif attack > defense + 10:
        combat_style = "swift striker with sleek offensive features"
        combat_features = "streamlined body, sharp edges"
    elif defense > attack + 30:
        combat_style = "fortress-like guardian with thick armor plating and protective shell"
        combat_features = "heavy armor, thick hide, protective spines"
    elif defense > attack + 10:
        combat_style = "sturdy defender with natural armor"
        combat_features = "tough skin, defensive posture"
    else:
        combat_style = "balanced fighter with versatile combat abilities"
        combat_features = "balanced proportions, adaptive features"
    
    # Speed-based characteristics
    if speed > 120:
        speed_traits = "lightning-fast with aerodynamic features, wind-swept design"
        speed_features = "streamlined fins, wing-like appendages"
    elif speed > 90:
        speed_traits = "very quick with sleek, aerodynamic build"
        speed_features = "lean limbs, flowing design"
    elif speed > 60:
        speed_traits = "moderately fast with agile proportions"
        speed_features = "proportioned limbs"
    else:
        speed_traits = "steady and deliberate with grounded, stable appearance"
        speed_features = "sturdy base, grounded stance"
    
    # HP-based vitality
    if hp > 150:
        vitality = "incredibly robust and overflowing with life energy"
        vitality_features = "glowing aura, vibrant colors, energetic pose"
    elif hp > 100:
        vitality = "very healthy and energetic"
        vitality_features = "bright colors, alert expression"
    elif hp > 70:
        vitality = "strong and vigorous"
        vitality_features = "confident stance, clear eyes"
    else:
        vitality = "nimble and quick"
        vitality_features = "alert posture, keen expression"
    
    # Special attack-based magical features
    if special_attack > 120:
        magic_traits = "radiating powerful magical energy with mystical auras and elemental effects"
        magic_features = "glowing markings, energy emanations, mystical symbols"
    elif special_attack > 90:
        magic_traits = "channeling moderate magical power"
        magic_features = "subtle glow, magical patterns"
    elif special_attack > 60:
        magic_traits = "showing some magical aptitude"
        magic_features = "faint magical hints"
    else:
        magic_traits = "primarily physical in nature"
        magic_features = "natural, non-magical appearance"
    
    # Special defense-based resistance features
    if special_defense > 120:
        resistance = "highly resistant to magical effects with protective enchantments"
        resistance_features = "shimmering protective barrier, anti-magic symbols"
    elif special_defense > 90:
        resistance = "moderately protected against magical attacks"
        resistance_features = "subtle protective markings"
    else:
        resistance = "naturally hardy"
        resistance_features = "natural resilience"
    
    # Experience-based sophistication
    if base_experience > 250:
        sophistication = "ancient and wise with battle-scarred veteran appearance"
    elif base_experience > 150:
        sophistication = "experienced and seasoned"
    else:
        sophistication = "youthful and energetic"
    
    # Color scheme based on stats
    if special_attack > attack:
        primary_colors = "mystical blues, purples, and ethereal whites"
    elif attack > 100:
        primary_colors = "fierce reds, oranges, and aggressive blacks"
    elif defense > 100:
        primary_colors = "earthy browns, greys, and protective greens"
    elif speed > 100:
        primary_colors = "electric yellows, sky blues, and swift silvers"
    else:
        primary_colors = "balanced natural tones with vibrant accents"
    
    # Create comprehensive prompt
    prompt = f"""A {sophistication}, {size} Pokemon creature that is a {combat_style}. 
    This {body_type} Pokemon is {build} and {speed_traits}, appearing {vitality}.
    It's {magic_traits} and {resistance}.
    
    Physical features: {combat_features}, {speed_features}, {vitality_features}, {magic_features}.
    Color scheme: {primary_colors}.
    Scale: {scale} Pokemon.
    
    Style: Official Pokemon artwork, anime style, cute yet powerful, highly detailed digital art, 
    vibrant colors, fantasy creature design, professional game art quality.
    No text, letters, or words in the image."""
    
    if progress_callback:
        progress_callback(30, "Creating detailed AI prompt...")
    
    try:
        # Load the Stable Diffusion pipeline
        if 'pokemon_pipeline' not in st.session_state:
            if progress_callback:
                progress_callback(40, "Loading Stable Diffusion model (first time only)...")
            
            st.session_state.pokemon_pipeline = DiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_safetensors=True
            )
            
            
            # Move to GPU if available, otherwise CPU
            if torch.cuda.is_available():
                st.session_state.pokemon_pipeline = st.session_state.pokemon_pipeline.to("cuda")
        
        if progress_callback:
            progress_callback(70, "Generating your Pokémon...")
        
        # Generate image
        with torch.inference_mode():
            result = st.session_state.pokemon_pipeline(
                prompt,
                num_inference_steps=25,  # More steps for better quality
                guidance_scale=8.0,     # Stronger guidance for better adherence to prompt
                height=512,
                width=512
            )
        
        if progress_callback:
            progress_callback(95, "Finalizing image...")
        
        image = result.images[0]
        return image, prompt
        
    except Exception as e:
        error_message = str(e)
        if "CUDA" in error_message or "GPU" in error_message:
            return None, f"GPU error (will try CPU): {error_message}"
        else:
            return None, f"Error generating image: {error_message}"
        
# This code defines a function to create models for predicting missing features in Pokémon data.
# It uses a Random Forest Regressor for each feature and imputes missing values using the median strategy.
# The `predict_missing` function takes user input and predicts missing features based on the trained models.
# The script can be run to demonstrate the functionality with an example user input.
models = make_models()
features = ['height','weight','base_experience','hp','attack','defense',
            'special-attack','special-defense','speed']
user_input = {}

st.title("Pokémon Feature Predictor")

# Sidebar navigation
tab1, tab2, tab3 = st.tabs(["Predictor - Make your Pokémon!", "Dataset", "Analytics"])

with tab1:
    st.header("Feature Predictor")
    st.write("Enter known values to predict missing Pokémon features with confidence intervals.")

    # Initialize predictions based on empty user input
    predictions = predict_missing(user_input, models)

    # Create main two-column layout: inputs left, predictions right
    col_input, col_predictions = st.columns([1, 1])

    with col_input:
        st.subheader("Input Features")
        
        # First, collect all user inputs
        for feature in features:
            value = st.number_input(
                f"Enter {feature.replace('-', ' ').title()}:",
                value=None,
                step=1.0,
                key=f"input_{feature}",
                help=f"Leave empty to see prediction for {feature}"
            )
            if value is not None and predictions[feature]['lower'] < value < predictions[feature]['upper']:
                user_input[feature] = value
            elif value is not None:
                st.warning(f"**{feature.replace('-', ' ').replace('_', ' ').title()}**: {value:.1f} is outside the prediction interval!")

    # Now get predictions based on all current inputs
    predictions = predict_missing(user_input, models)

    with col_predictions:
        st.subheader("Live Predictions")
        
        # Display each feature with its prediction info
        for feature in features:
            pred_data = predictions[feature]
            
            if feature in user_input:
                # User provided input
                st.success(f"**{feature.replace('-', ' ').replace('_', ' ').title()}**: {user_input[feature]:.1f} (Your input)")
            else:
                # Show prediction
                mean_val = pred_data['mean']
                lower_val = pred_data['lower']
                upper_val = pred_data['upper']
            
                # Create a box plot for the confidence interval
                fig, ax = plt.subplots(figsize=(6, 0.2))
                
                # Draw a continuous line from lower bound to upper bound
                y_pos = 1
                ax.plot([lower_val, upper_val], [y_pos, y_pos], 'k-', linewidth=1, label='95% CI')
                
                # Add vertical caps at the ends
                cap_height = 0.1
                ax.plot([lower_val, lower_val], [y_pos - cap_height*0.5, y_pos + cap_height*0.5], 'k-', linewidth=1.5)
                ax.plot([upper_val, upper_val], [y_pos - cap_height*0.5, y_pos + cap_height*0.5], 'k-', linewidth=1.5)
                
                # Add red line for the mean
                ax.plot([mean_val, mean_val], [y_pos - cap_height, y_pos + cap_height], 'r-', linewidth=2, label='Mean')
                
                # Set labels and formatting
                ax.set_xlim(lower_val - (upper_val - lower_val) * 0.1, 
                           upper_val + (upper_val - lower_val) * 0.1)
                ax.set_xlabel(f'{feature.replace("-", " ").replace("_", " ").title()} Value')
                ax.set_yticks([])
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                ax.text(lower_val, 1.3, f'{lower_val:.1f}', ha='center', va='bottom', fontsize=9)
                ax.text(mean_val, 1.3, f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9, weight='bold')
                ax.text(upper_val, 1.3, f'{upper_val:.1f}', ha='center', va='bottom', fontsize=9)
        
                st.pyplot(fig)
                plt.close(fig)
    
    total_input = sum(1 for v in user_input.values() if v is not None)

    if total_input == 9:
        
        # Initialize session state for button styling
        if 'last_clicked' not in st.session_state:
            st.session_state.last_clicked = 'image'  # Default to image

        # Create two columns for different generation options
        col1, col2 = st.columns(2)
        
        with col1:
            generate_image = st.button(
                "Generate Image", 
                use_container_width=True
            )
        
        with col2:
            generate_profile = st.button(
                "Generate Name", 
                use_container_width=True
            )
        
        # Option to generate both
        generate_both = st.button("Generate Complete Pokémon (Image + Profile)", use_container_width=True)

        if generate_image:
            st.write("**Generating your custom Pokémon image...**")
            
            # Create progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Define progress callback function
            def update_progress(percent, message):
                progress_bar.progress(percent / 100.0)  # Convert percentage to 0.0-1.0 range
                status_text.text(message)
            
            # Generate Pokémon image with progress updates
            img, prompt = generate_pokemon_image_free(user_input, update_progress)
            
            if img:
                # Final progress update
                progress_bar.progress(1.0)  # 100% completion
                status_text.text("Your Pokémon is ready!")
                
                # Small delay to show completion
                import time
                time.sleep(0.5)
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                # Show the image
                st.image(img, caption="Generated Pokémon Image", use_container_width=True)
                
                # Display prompt in a cleaner, expandable format
                with st.expander("View AI Prompt Details", expanded=False):
                    st.text_area("Full prompt used to generate this Pokemon:", prompt, height=150, disabled=True)
                
            else:
                # Error occurred
                progress_bar.empty()
                status_text.empty()
                st.error(f"{prompt}")  # prompt contains error message if img is None
                st.info("💡 **Tip:** The first run might be slower as it downloads the AI model. Try again!")
        
        elif generate_profile:
            st.write("**Generating your Pokémon name...**")
            
            # Create progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Define progress callback function
            def update_progress(percent, message):
                progress_bar.progress(percent / 100.0)  # Convert percentage to 0.0-1.0 range
                status_text.text(message)
            
            # Generate Pokémon profile
            profile = generate_pokemon_name(user_input, update_progress)

            
            # Final progress update
            progress_bar.progress(1.0)  # 100% completion
            status_text.text("Your Pokémon profile is ready!")
            
            # Small delay to show completion
            import time
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display the generated profile
            st.success(f"Meet your new Pokémon: **{profile['name']}**!")
            
           
            st.subheader(f"{profile['name']} Profile")
            st.write(f"**Suggested Type:** {profile['type_suggestion']}")
            st.write(f"**Specialty:** {profile['stats_summary']}")
            
            # Show name generation details
            with st.expander("View Name Generation Details", expanded=False):
                if 'name_prompt' in profile:
                    st.text_area("Name generation prompt:", profile['name_prompt'], height=80, disabled=True)
                if 'error' in profile:
                    st.warning(f"Note: Some features used fallback generation due to: {profile['error']}")
        
        elif generate_both:
            st.write("**Creating your complete Pokémon...**")
            
            # Create progress bar and status text
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Define progress callback function
            def update_progress(percent, message):
                progress_bar.progress(percent / 100.0)  # Convert percentage to 0.0-1.0 range
                status_text.text(message)
            
            # Generate profile first (faster)
            status_text.text("📝 Generating name...")
            progress_bar.progress(0.1)  # 10%
            profile = generate_pokemon_name(user_input, lambda p, m: update_progress(10 + p//3, m))
            
            # Then generate image
            status_text.text("🎨 Now generating image...")
            progress_bar.progress(0.4)  # 40%
            img, prompt = generate_pokemon_image_free(user_input, lambda p, m: update_progress(40 + p*0.6, m))
            
            # Final progress update
            progress_bar.progress(1.0)  # 100% completion
            status_text.text("Your complete Pokémon is ready!")
            
            # Small delay to show completion
            import time
            time.sleep(0.5)
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            if img:
                # Display complete Pokémon
                st.success(f"Meet your new Pokémon: **{profile['name']}**!")
                
                # Show image first
                st.image(img, caption=f"{profile['name']} - Your Generated Pokémon", use_container_width=True)
                
                # Show profile below image
                st.subheader(f"{profile['name']} Profile")
                st.write(f"**Suggested Type:** {profile['type_suggestion']}")
                st.write(f"**Specialty:** {profile['stats_summary']}")
                    
                
                # Expandable prompt details
                with st.expander("View AI Generation Details", expanded=False):
                    st.text_area("Image generation prompt:", prompt, height=100, disabled=True)
                    if 'name_prompt' in profile:
                        st.text_area("Name generation prompt:", profile['name_prompt'], height=80, disabled=True)
                    if 'error' in profile:
                        st.warning(f"Text generation note: {profile['error']}")
            else:
                st.error(f"Image generation failed: {prompt}")
                # Still show the profile even if image failed
                st.success(f"But here's your Pokémon profile: **{profile['name']}**!")
                

                st.write(f"**Type:** {profile['type_suggestion']}")
                st.write(f"**Specialty:** {profile['stats_summary']}")
                
                # Show name generation details even when image fails
                with st.expander("View Name Generation Details", expanded=False):
                    if 'name_prompt' in profile:
                        st.text_area("Name generation prompt:", profile['name_prompt'], height=80, disabled=True)
                    if 'error' in profile:
                        st.warning(f"Note: {profile['error']}")
                

with tab2:
    # Dataset Explorer
    st.header("Dataset Explorer")
    
    st.subheader("Full Pokemon Dataset")
    df_new = df.drop(columns=['type_list', 'ability_list'])
    
    st.write(f"Dataset contains {len(df_new)} Pokémon with {len(df_new.columns)} features:")
    st.dataframe(df_new, use_container_width=True)
    
    st.write("**Data Source:** [PokéAPI](https://pokeapi.co/)")
    
    # Basic statistics
    st.subheader("Dataset Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Numerical Features Summary:**")
        numeric_cols = df_new.select_dtypes(include=[np.number]).columns
        st.dataframe(df_new[numeric_cols].describe())
    
    with col2:
        st.write("**Sample Pokémon:**")
        sample_pokemon = df_new.sample(5)
        for idx, row in sample_pokemon.iterrows():
            st.write(f"**{row['name'].title()}**")
            st.write(f"Height: {row['height']}, Weight: {row['weight']}, HP: {row['hp']}")


with tab3:
    # Model Analytics
    st.header("Model Analytics")
    st.write("Analysis of the Random Forest models used for prediction.")
    
    # Model information
    st.subheader("Model Details")
    st.write("- **Algorithm:** Random Forest Regressor")
    st.write("- **Number of Trees:** 100 per feature")
    st.write("- **Features Predicted:** 9 (height, weight, base_experience, hp, attack, defense, special-attack, special-defense, speed)")
    st.write("- **Missing Value Strategy:** Median imputation")
    st.write("- **Confidence Interval:** 95% (±1.96 standard deviations)")
    
    # Feature importance (for one model as example)
    st.subheader("Feature Importance Example")
    st.write("*Showing feature importance for predicting 'attack' values:*")
    
    sample_model = models['attack']
    feature_names = sample_model.feature_names_in_
    importances = sample_model.named_steps['regressor'].feature_importances_
    
    # Verify alignment - show the mapping
    st.write("**Feature-Importance Mapping Verification:**")

    col1, col2 = st.columns(2)
    for i, (feature, importance) in enumerate(zip(feature_names, importances)):
        with col1 if i % 2 == 0 else col2:
            st.write(f"**{feature.replace('-', ' ').replace('_', ' ').title()} =** {importance:.4f}")

    importance_df = pd.DataFrame({
        'Feature': [f.replace('-', ' ').replace('_', ' ').title() for f in feature_names], 
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    # Show the raw data table for verification
    st.write("**Raw Feature Importance Data:**")
    st.dataframe(importance_df, hide_index=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(importance_df['Feature'], importance_df['Importance'])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance for Predicting Attack')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


