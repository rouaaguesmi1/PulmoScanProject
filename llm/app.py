import os
import logging
from flask import Flask, render_template, request, jsonify
import google.generativeai as genai
from dotenv import load_dotenv

# --- Configuration ---
load_dotenv()  # Load environment variables from .env file
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file. Please create .env and add your key.")

# Configure the Gemini API client
try:
    genai.configure(api_key=API_KEY)
    # Choose the appropriate Gemini model
    # 'gemini-pro' is a good starting point for text generation
    model = genai.GenerativeModel('gemini-2.0-flash')
except Exception as e:
    logging.error(f"Error configuring Gemini API: {e}")
    # Handle configuration error gracefully if needed, maybe exit or use a fallback
    raise

# Configure Flask App
app = Flask(__name__)
app.secret_key = os.urandom(24) # For potential future session use

# --- Prompt Engineering ---

BASE_PROMPT = """
You are PneumaAssistant, an AI assistant specialized *exclusively* in the field of lung cancer.
Your knowledge is strictly limited to lung cancer, its diagnosis, types, stages, treatments (like surgery, chemotherapy, radiation, immunotherapy, targeted therapy), side effects, prognosis, prevention, screening, related lung conditions, and supportive care.
You MUST NOT discuss any other medical conditions, general health topics, or any non-medical subjects.
If asked about anything outside the scope of lung cancer, you must politely decline and state that your expertise is solely focused on lung cancer.
Do not provide specific medical advice, diagnosis, or treatment plans. Always recommend consulting with qualified healthcare professionals for personal medical concerns.
"""

PATIENT_PROMPT_EXTENSION = """
You are interacting with a patient or their loved one.
Your tone must be: Empathetic, compassionate, supportive, patient, and understanding.
Use clear, simple language. Avoid overly technical jargon.
Focus on providing general information, explaining concepts gently, offering emotional support, and pointing towards reliable resources (without giving specific medical advice).
Be encouraging and hopeful where appropriate, while remaining realistic.
Acknowledge the emotional difficulty of dealing with lung cancer.
"""

DOCTOR_PROMPT_EXTENSION = """
You are interacting with a healthcare professional (e.g., doctor, oncologist, nurse).
Your tone must be: Professional, precise, objective, and informative.
Use accurate medical terminology.
Provide detailed, evidence-based information based on current medical literature and clinical guidelines related to lung cancer.
You can discuss complex topics like specific treatment protocols, clinical trial data (if available in your knowledge base), molecular subtypes, diagnostic nuances, and comparative effectiveness, but always within the bounds of generally accepted medical knowledge.
Stick strictly to the facts and avoid speculation.
"""

def create_prompt(user_query: str, mode: str) -> str:
    """Builds the final prompt for the Gemini API based on the user mode."""
    if mode == "patient":
        mode_specific_prompt = PATIENT_PROMPT_EXTENSION
    elif mode == "doctor":
        mode_specific_prompt = DOCTOR_PROMPT_EXTENSION
    else:
        # Default or fallback - perhaps a neutral but still focused prompt
        mode_specific_prompt = "Provide accurate information about lung cancer in a clear manner."

    # Combine prompts - ensuring the user query is clearly delineated
    full_prompt = f"{BASE_PROMPT}\n\n{mode_specific_prompt}\n\nUser Query:\n\"\"\"\n{user_query}\n\"\"\""
    return full_prompt

# --- Flask Routes ---

@app.route('/')
def index():
    """Renders the main chat interface."""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handles chat requests from the frontend."""
    try:
        data = request.get_json()
        user_query = data.get('query')
        mode = data.get('mode') # 'patient' or 'doctor'

        if not user_query or not mode:
            return jsonify({'error': 'Missing query or mode in request'}), 400

        # --- Safety Check (Basic) ---
        # You might add more robust checks here if needed
        if len(user_query) > 2000: # Limit query length
             return jsonify({'error': 'Query too long'}), 400

        # --- Generate Prompt ---
        prompt = create_prompt(user_query, mode)
        # print(f"--- Sending Prompt to Gemini ---\n{prompt}\n-------------------------------") # For debugging

        # --- Call Gemini API ---
        # Add safety settings to filter harmful content
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        ]

        try:
            response = model.generate_content(
                prompt,
                safety_settings=safety_settings
                # You can add generation_config here for temperature, top_k, etc.
                # generation_config=genai.types.GenerationConfig(temperature=0.7)
            )

            # --- Process Response ---
            # Check for blocked content due to safety settings
            if not response.candidates:
                 assistant_response = "I cannot provide a response to this request due to safety guidelines."
            else:
                # Accessing the text might vary slightly depending on API version/response structure
                # Check response object structure if issues arise (e.g., print(response))
                assistant_response = response.text

        except Exception as e:
            logging.error(f"Error generating content with Gemini: {e}")
            # Check for specific API errors if needed (e.g., quota limits, invalid requests)
            if "API key not valid" in str(e):
                 assistant_response = "Error: Invalid API Key. Please check configuration."
            else:
                 assistant_response = f"Sorry, I encountered an error trying to generate a response. Error: {e}"
            return jsonify({'error': assistant_response}), 500 # Internal Server Error

        # --- Return Response ---
        return jsonify({'response': assistant_response})

    except Exception as e:
        logging.exception("An unexpected error occurred in /chat endpoint.") # Log the full traceback
        return jsonify({'error': f'An unexpected server error occurred: {e}'}), 500


# --- Run Application ---
if __name__ == '__main__':
    # Set debug=False for production deployment
    app.run(debug=True, host='0.0.0.0', port=5000)