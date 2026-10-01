import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "Pakeezafaryad-hf/Plagiarism-Checker"

st.set_page_config(
    page_title="Mini Plagiarism Checker",
    page_icon="🔎"
)

st.title("🔎 Mini Plagiarism Checker")
st.write("Compare two pieces of text and check whether they are similar.")


@st.cache_resource
def load_model():

    hf_token = st.secrets["HF_TOKEN"] if "HF_TOKEN" in st.secrets else None

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=hf_token
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_ID,
        token=hf_token
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


def predict(text1, text2, tokenizer, model, device):

    input_text = f"sentence1: {text1} sentence2: {text2}"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=128,
        truncation=True
    ).to(device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=8
        )

    result = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip().lower()

    # Debug: show the model's actual output
    st.write("Debug - Model output:", result)

    # Model's actual labels
    if "not duplicate" in result:
        return "Not similar"

    elif "duplicate" in result:
        return "Similar text detected"

    else:
        return f"Model output: {result}"


text1 = st.text_area(
    "Enter original text",
    height=150
)

text2 = st.text_area(
    "Enter text to compare",
    height=150
)


if st.button("🔍 Check Similarity"):

    if not text1.strip() or not text2.strip():

        st.warning("Please enter both texts.")

    else:

        with st.spinner("Checking similarity..."):

            tokenizer, model, device = load_model()

            result = predict(
                text1,
                text2,
                tokenizer,
                model,
                device
            )

        if result == "Similar text detected":

            st.warning("⚠️ Similar text detected")

        elif result == "Not similar":

            st.success("✅ Not similar")

        else:

            st.info(f"ℹ️ {result}")


st.caption(
    "This tool detects text similarity; it does not by itself prove plagiarism."
)
