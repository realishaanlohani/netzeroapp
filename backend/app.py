import os
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# --- Configuration ---

# Find the .env file in the 'keys' subdirectory
# Get the directory where this script is running
base_dir = os.path.dirname(os.path.abspath(__file__))
# Define the path to the .env file
dotenv_path = os.path.join(base_dir, 'keys', '.env')

# Load environment variables (your API key) from the specified path
load_dotenv(dotenv_path=dotenv_path)

# Initialize Flask app
app = Flask(__name__)
# Enable CORS to allow your frontend to talk to your backend
CORS(app)

# Configure the Gemini API
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    # Give a more specific error message if the key isn't found
    raise ValueError(f"GEMINI_API_KEY not found. Please ensure it is set in the {dotenv_path} file.")

genai.configure(api_key=api_key)

# Select the model.
# Updated to gemini-2.5-flash-preview-09-2025 as a robust flash model.
model = genai.GenerativeModel("gemini-2.5-flash-preview-09-2025")

# --- Constants for Calculation ---

# Simplified database of GPU Thermal Design Power (TDP) in Watts
# Add more models as needed
GPU_TDP_DB = {
    "NVIDIA H100 (SXM)": 700,
    "NVIDIA H100 (PCIe)": 350,
    "NVIDIA A100 (SXM)": 400,
    "NVIDIA A100 (PCIe)": 250,
    "NVIDIA RTX 4090": 450,
    "NVIDIA RTX 3090": 350,
    "Google TPU v4": 275,
    "Other/Unknown": 300,  # A generic default
}

# Average Power Usage Effectiveness (PUE) for data centers
# 1.0 is perfect, 1.5 is a reasonable average.
PUE = 1.5

# Average Carbon Intensity of the grid (kg CO2e per kWh)
# This varies *wildly* by location. 0.39 is a recent global-ish average.
CARBON_INTENSITY_KG_PER_KWH = 0.39

# --- API Endpoint ---


@app.route("/calculate", methods=["POST"])
def calculate_emissions():
    try:
        # 1. Get data from the frontend JSON
        data = request.json

        user_type = data.get("userType")
        gpu_model = data.get("gpuModel")
        num_gpus = int(data.get("numGpus", 1))
        duration_hours = float(data.get("durationHours", 0))

        # Get dataset size (for the Gemini prompt)
        dataset_size = data.get("datasetSize", "an unknown")

        # 2. Perform the calculation
        gpu_tdp_watts = GPU_TDP_DB.get(gpu_model, 300)  # Default to 300W if not found

        # Energy (kWh) = (Power in Watts * Num GPUs * Time in Hours * PUE) / 1000 (to convert W to kW)
        total_energy_kwh = (gpu_tdp_watts * num_gpus * duration_hours * PUE) / 1000

        # Carbon (kg CO2e) = Energy * Carbon Intensity
        total_carbon_kg = total_energy_kwh * CARBON_INTENSITY_KG_PER_KWH

        # 3. Handle Corporation AI Suggestions
        gemini_suggestions = None
        if user_type == "corporation":
            # Create a prompt for the Gemini API
            prompt = f"""
            You are an expert in Green AI and sustainable computing.
            A corporation is training an AI model with the following parameters:
            - GPU Model: {gpu_model}
            - Number of GPUs: {num_gpus}
            - Training Duration: {duration_hours} hours
            - Dataset Size: {dataset_size}

            This training run consumed an estimated {total_energy_kwh:.2f} kWh of energy
            and produced an estimated {total_carbon_kg:.2f} kg of CO2e.

            Please provide 3-5 concise, actionable recommendations for this corporation
            to reduce its power consumption and carbon footprint for future AI training.
            Focus on practical steps like model optimization (e.g., quantization, pruning),
            hardware selection (e.g., newer-gen GPUs, TPUs),
            data center PUE, and choosing training locations with low-carbon energy grids.
            Format the response for a <pre> tag, using line breaks.
            """

            try:
                # The generate_content call is correct.
                # The issue was likely the API key not loading.
                response = model.generate_content(prompt)
                gemini_suggestions = response.text
            except Exception as e:
                print(f"Gemini API error: {e}")
                gemini_suggestions = "Error generating AI suggestions. Please check backend logs and API key."

        # 4. Send the result back to the frontend
        return jsonify(
            {
                "energy_kwh": total_energy_kwh,
                "carbon_kg": total_carbon_kg,
                "suggestions": gemini_suggestions,
            }
        )

    except Exception as e:
        print(f"Server error: {e}")
        return jsonify({"error": str(e)}), 500


# --- Run the App ---

if __name__ == "__main__":
    # Runs the server on http://127.0.0.1:5000
    app.run(debug=True, port=5000)