import os
from dotenv import load_dotenv
load_dotenv()

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
import torch
import streamlit as st

# === TOGGLE YOUR MODEL ===
USE_TINYLLaMA = True
USE_FLAN_T5 = False
USE_GEMINI = False

# === TinyLlama ===
if USE_TINYLLaMA:
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    def load_llm():
        if "tinyllama_model" not in st.session_state:
            print("🧠 Loading TinyLlama model...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                device_map="auto",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            st.session_state.tinyllama_model = {"tokenizer": tokenizer, "model": model}
            print("✅ TinyLlama loaded.")
        return st.session_state.tinyllama_model

    def generate_answer(llm, prompt: str, max_tokens=200):
        tokenizer = llm["tokenizer"]
        model = llm["model"]

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=0.7,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        return tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, "").strip()

# === FLAN-T5 ===
elif USE_FLAN_T5:
    def load_llm():
        if "flan_model" not in st.session_state:
            print("🧠 Loading FLAN-T5...")
            model_name = "google/flan-t5-large"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            device = 0 if torch.cuda.is_available() else -1
            pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device)
            st.session_state.flan_model = pipe
            print("✅ FLAN-T5 loaded.")
        return st.session_state.flan_model

    def generate_answer(llm, prompt: str, max_tokens=300):
        tokenizer = llm["tokenizer"]
        model = llm["model"]

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        output = model.generate(
        input_ids,
        max_new_tokens=max_tokens,
        temperature=0.7,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

        full_text = tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, "").strip()

    # ✅ Return only the first sentence or answer block
        if "Answer:" in full_text:
            full_text = full_text.split("Answer:")[-1]

        return full_text.strip().split("\n")[0]


# === Gemini (requires API key) ===
elif USE_GEMINI:
    import google.generativeai as genai

    def load_llm():
        if "gemini_model" not in st.session_state:
            print("🌐 Loading Gemini-Pro model...")
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            model = genai.GenerativeModel("models/gemini-pro")
            st.session_state.gemini_model = model
            print("✅ Gemini loaded.")
        return st.session_state.gemini_model

    def generate_answer(model, prompt: str, max_tokens=300):
        response = model.generate_content(prompt)
        return response.text.strip()
