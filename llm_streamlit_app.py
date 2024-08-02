import json
import requests
import streamlit as st
import fitz  # PyMuPDF

# API Key dan Endpoint
API_KEY = '55f9e8ca-f675-49f5-93b6-ecdeec9d538c'
API_ENDPOINT = 'https://saas.cakra.ai/genv2/llms'

# Fungsi untuk memuat dokumen PDF
def load_pdf(file):
    pdf_document = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

# Fungsi untuk mengirimkan request ke API LLM
def get_llm_response(prompt):
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {API_KEY}'
    }
    data = {
        "model_name": "brain-v2",
        "messages": [
            {"role": "system", "content": "Your Chatbot AI Assistant"},
            {"role": "user", "content": prompt}
        ],
        "max_new_tokens": 100,
        "do_sample": False,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 1.0
    }
    response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        return response.json()
    else:
        return {"choices": [{"content": "Maaf, terjadi kesalahan dalam memproses permintaan Anda."}]}

# Fungsi untuk mengekstraksi jawaban dari respon API
def extract_answer(response):
    if "choices" in response:
        return response['choices'][0]['content']
    else:
        return "Tidak ada jawaban yang tersedia."

# Inisialisasi session state untuk menyimpan riwayat pertanyaan dan jawaban
if 'history' not in st.session_state:
    st.session_state.history = []

# Streamlit UI
st.title('Tanya Jawab Interaktif dengan LLM')
st.write('Upload dokumen PDF untuk diproses oleh LLM:')

uploaded_file = st.file_uploader("Upload dokumen PDF", type=["pdf"])

if uploaded_file is not None:
    document_content = load_pdf(uploaded_file)

    # Input untuk pertanyaan pengguna dengan text_area
    user_question = st.text_area("Masukkan pertanyaan Anda berdasarkan dokumen di atas:")

    if st.button("Tanya"):
        if user_question.strip():  # Pastikan pertanyaan tidak kosong atau hanya spasi
            # Mengirimkan pertanyaan pengguna ke LLM
            response = get_llm_response(user_question)
            answer = extract_answer(response)
            
            # Simpan pertanyaan dan jawaban ke dalam riwayat
            st.session_state.history.append({"question": user_question, "answer": answer})
            
            st.write(f"**Jawaban:** {answer}")
        else:
            st.warning("Silakan masukkan pertanyaan terlebih dahulu.")

# Menampilkan riwayat pertanyaan dan jawaban
st.write("## Riwayat Pertanyaan dan Jawaban:")
if st.session_state.history:
    for idx, entry in enumerate(st.session_state.history, 1):
        st.write(f"**Pertanyaan {idx}:** {entry['question']}")
        st.write(f"**Jawaban {idx}:** {entry['answer']}")
else:
    st.write("Belum ada pertanyaan yang diajukan.")
