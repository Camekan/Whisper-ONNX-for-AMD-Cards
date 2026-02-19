🎙️ Whisper ONNX Transcriber Setup

You can easily run Whisper ONNX locally using this script. Follow these simple steps to get started:
📥 1. Download the AI Models

Create a new folder for your project. Then, go to the Hugging Face repository and download the following three files directly into that folder: https://huggingface.co/onnx-community/whisper-large-v3-turbo/tree/main/onnx

    encoder_model.onnx

    encoder_model.onnx_data

    decoder_with_past_model.onnx

🐍 2. Download the Project Files

Next, download the core project files and place them into the same folder as your AI models:

    31.py (The main Python script)

    whisperonnx.bat (The launcher file)

    requirements.txt (The dependencies list)

⚙️ 3. Install Dependencies

    Open Command Prompt as an Administrator.

    Navigate to your project folder using the cd command (e.g., cd C:\Path\To\Your\Folder).

    Create a venv environment with: python -m venv venv

    Activate the venv environment: venv\Scripts\activate

    Then, install all necessary Python packages by running:

    pip install -r requirements.txt

🚀 4. Final Setup & Launch

Once the installation is complete, right-click the whisperonnx.bat file and select Edit. Update the path inside the file to match your actual folder location and save it.

You are all set! You can now double-click whisperonnx.bat from anywhere on your computer. It will automatically launch the 31.py script and open the transcriber interface directly in your web browser.
