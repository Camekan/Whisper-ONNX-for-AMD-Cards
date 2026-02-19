"""
Whisper ONNX Transcriber - The Ultimate Hybrid
✅ Crash-proof granular phrase chunks (~15s)
✅ Extended multi-language & auto-detect support
✅ DummyWhisperModel config patch applied
✅ Bulletproof NoneType handler
"""

import librosa
import numpy as np
import os
import gradio as gr
from transformers import AutoProcessor, pipeline
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
from huggingface_hub import hf_hub_download

print("="*70)
print("🎙️ Whisper ONNX Transcriber - Master Version")
print("="*70)

# Download config files
for filename in ["config.json", "generation_config.json", "preprocessor_config.json"]:
    if not os.path.exists(filename):
        hf_hub_download(repo_id="openai/whisper-large-v3-turbo", filename=filename, local_dir=".")

# 1. Load the Processor
processor = AutoProcessor.from_pretrained("openai/whisper-large-v3-turbo")

# 2. Load the ONNX Model with DirectML (AMD GPU)
print("\n⚡ Loading ONNX model with DirectML (AMD GPU)...")
model = ORTModelForSpeechSeq2Seq.from_pretrained(
    ".", 
    encoder_file_name="encoder_model.onnx",                 
    decoder_file_name="decoder_model.onnx",                     
    decoder_with_past_file_name="decoder_with_past_model.onnx", 
    use_merged=False,
    provider="DmlExecutionProvider"
)

# 🛑 THE BUG FIX: Inject the missing config into the Optimum dummy model
if hasattr(model, "model") and not hasattr(model.model, "config"):
    model.model.config = model.config

# 3. Create the Hugging Face Pipeline
transcriber = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    chunk_length_s=15,       # ✅ FIX: Keeps the timestamps granular without crashing!
    stride_length_s=[3, 1],  
    batch_size=1,
)

def transcribe_audio(audio_file, language, include_timestamps):
    """Gradio interface function"""
    if audio_file is None:
        return "❌ Please upload an audio file."
    
    try:
        audio_array, sr = librosa.load(audio_file, sr=16000)
        audio_array = np.pad(audio_array, (0, 16000), mode='constant')
        
        # ✅ CLAUDE'S FEATURE: Map the UI languages to ISO codes
        lang_map = {
            "English": "en",
            "Turkish": "tr",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
            "Italian": "it",
            "Portuguese": "pt",
            "Auto-detect": None
        }
        lang_code = lang_map.get(language)
        
        print(f"🔄 Transcribing ({language})...")
        
        # Build generation kwargs (leaves out language if Auto-detect is chosen)
        gen_kwargs = {
            "task": "transcribe",
            "max_new_tokens": 440
        }
        if lang_code:
            gen_kwargs["language"] = lang_code
        
        result = transcriber(
            audio_array, 
            return_timestamps=include_timestamps, 
            generate_kwargs=gen_kwargs
        )
        
        if include_timestamps and "chunks" in result:
            formatted_text = ""
            last_end = 0.0
            previous_texts = set()
            
            for chunk in result["chunks"]:
                # 🛑 BULLETPROOF PARSER
                ts = chunk.get("timestamp")
                if ts is None:
                    start, end = last_end, last_end + 3.0
                else:
                    start = ts[0] if ts[0] is not None else last_end
                    end = ts[1] if ts[1] is not None else start + 3.0
                    
                if end < start:
                    continue
                    
                text = chunk.get("text", "").strip()
                if not text:
                    continue
                    
                # Overlap filter
                if text in previous_texts and start < last_end + 0.5:
                    continue
                    
                # Format perfectly as [00.00s → 05.00s]
                formatted_text += f"[{start:05.2f}s → {end:05.2f}s] {text}\n"
                last_end = max(last_end, end)
                previous_texts.add(text)
                
            return formatted_text.strip()
        else:
            return result.get("text", "").strip()
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}"

# Create Gradio interface
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🎙️ Whisper Large V3 Turbo - Granular Timestamps
        
        ### ⚡ AMD GPU DirectML + Crash-Proof Chunking
        """
    )
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="🎤 Audio Input")
            
            language_dropdown = gr.Dropdown(
                choices=["English", "Turkish", "Spanish", "French", "German", "Italian", "Portuguese", "Auto-detect"],
                value="Auto-detect", 
                label="🌍 Language"
            )
            
            timestamp_checkbox = gr.Checkbox(
                value=True,
                label="📍 Include Timestamps",
                info="Show timestamps for every spoken phrase"
            )
            
            transcribe_btn = gr.Button("🚀 Transcribe", variant="primary", size="lg")
        
        with gr.Column():
            output_text = gr.Textbox(label="📝 Transcription", lines=20)
            
    transcribe_btn.click(fn=transcribe_audio, inputs=[audio_input, language_dropdown, timestamp_checkbox], outputs=output_text)

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft(), inbrowser=True)
