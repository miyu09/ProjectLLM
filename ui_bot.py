import os
import glob
import time
import warnings
import streamlit as st

# 🧹 ซ่อนแจ้งเตือนสีเหลืองกวนใจ
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_ollama import OllamaLLM, OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ==========================================
# ตั้งค่าระบบ (Config)
# ==========================================
MODEL_NAME = "qwen2.5:3b"        # ใช้ Qwen 3B (ความจุระดับกลาง ฉลาดและเร็วพอดีกับ VRAM 4GB)
EMBED_MODEL = "bge-m3"           # โมเดลสมองส่วนจำ (ตัวแปลงภาษา)

# Page Config
st.set_page_config(
    page_title="Medical Assessment RAG Bot",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .time-badge {
        background-color: #E2E8F0;
        color: #0F172A;
        padding: 5px 12px;
        border-radius: 8px;
        font-size: 0.88rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_rag_chain():
    """โหลดและสร้าง RAG Chain พร้อมแคชไว้ในหน่วยความจำเพื่อความรวดเร็ว"""
    data_files = [f for f in glob.glob("data/*.*") if f.endswith(('.txt', '.md')) and not f.endswith("run.txt")]
    if not data_files:
        return None, "ไม่พบไฟล์ข้อมูล (.txt / .md) ในโฟลเดอร์ data"

    documents = []
    loaded_files = []
    for file_path in data_files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
            loaded_files.append(os.path.basename(file_path))
        except Exception as e:
            pass

    if not documents:
        return None, "ไม่สามารถอ่านไฟล์ในโฟลเดอร์ data ได้"

    CHROMA_PATH = "./chroma_db"
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) # ลด chunk size ลงเพื่อให้ประมวลผลเร็วขึ้น
        chunks = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.0,
        num_ctx=8192           # เพิ่มบริบทให้กว้างขึ้น เพื่อป้องกันปัญหาคำสั่งถูกตัดเมื่อดึงข้อมูลหลายชิ้น (k=5)
    )

    # โหลดเกณฑ์การประเมิน (Grading) เข้าสู่ System Prompt ถาวร
    grading_criteria = ""
    try:
        with open("data/Grading.md", "r", encoding="utf-8") as f:
            grading_criteria = f.read()
    except Exception as e:
        grading_criteria = "ไม่พบไฟล์ Grading.md"

    system_template = f"""You are a medical AI assistant. Your task is to assess symptoms and answer questions strictly based on the provided reference context.
CRITICAL INSTRUCTIONS:
1. YOU MUST ANSWER ENTIRELY IN THAI LANGUAGE. Do NOT use any Chinese characters under any circumstances. If you encounter English medical terms, keep them in English or translate them to Thai. DO NOT output any Chinese.
2. Only use the provided "Reference Context" to assess the severity grade.
3. For the "คำแนะนำเบื้องต้นสำหรับคนไข้" section, you may use your general medical knowledge to provide appropriate and safe advice.
4. Your response MUST follow this exact format:

ประเมินระดับความรุนแรง:
(คุณต้อง "ประเมินและเลือก" Grade ที่ตรงกับอาการของผู้ป่วยมากที่สุด "เพียง 1 ระดับเท่านั้น". If the exact symptom is not explicitly listed, ESTIMATE the closest Grade based on the context. ห้ามคัดลอก Grade อื่นๆ มาแสดง พร้อมอธิบายเหตุผลสั้นๆ)

คำแนะนำเบื้องต้นสำหรับคนไข้:
(Provide the patient advice entirely in Thai)
4. Highlight any dangerous symptoms found in the data.

=== GRADING CRITERIA ===
{grading_criteria}
======================="""

    human_template = """Reference Context:
{context}

Question: {question}"""

    PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="[ไฟล์: {source}]\n{page_content}"
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}), # ปรับเป็น 3 ชิ้น เพื่อลดภาระให้อ่านน้อยลงและเร็วขึ้น
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT,
            "document_prompt": document_prompt
        }
    )

    return qa_chain, loaded_files

# --- Header ---
st.markdown('<div class="main-header">🏥 Medical Assessment RAG Bot</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ระบบประเมินอาการและตอบคำถามทางการแพทย์ด้วย RAG (พร้อมตัวจับเวลาประมวลผล)</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ สถานะและข้อมูลระบบ")
    st.write(f"🤖 **LLM Model:** `{MODEL_NAME}`")
    st.write(f"🧠 **Embedding Model:** `{EMBED_MODEL}`")
    
    st.divider()
    
    qa_chain, loaded_files = load_rag_chain()
    
    st.subheader("📚 เอกสารที่อ่านเข้าระบบ")
    if isinstance(loaded_files, list):
        for f in loaded_files:
            st.success(f"📄 {f}")
    else:
        st.error(loaded_files)

    st.divider()
    if st.button("🔄 โหลด Vector DB ใหม่"):
        import shutil
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        st.cache_resource.clear()
        st.rerun()

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display Messages ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "rag_time" in message:
            st.markdown(f'<div class="time-badge">⏱️ เวลาประมวลผล RAG: {message["rag_time"]:.2f} วินาที</div>', unsafe_allow_html=True)
        if "sources" in message and message["sources"]:
            with st.expander(f"🔍 ดูข้อมูลอ้างอิงที่ AI ดึงมาอ่าน ({len(message['sources'])} ชิ้น)"):
                for i, doc in enumerate(message["sources"]):
                    source_file = os.path.basename(doc.metadata.get('source', 'ไม่ทราบไฟล์'))
                    st.markdown(f"**ชิ้นที่ {i+1}** (จากไฟล์: `{source_file}`)")
                    st.code(doc.page_content, language="markdown")

# --- User Input ---
if prompt := st.chat_input("พิมพ์คำถามอาการป่วย หรือสอบถามเกณฑ์การประเมินที่นี่..."):
    # Add User Message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process AI Response
    with st.chat_message("assistant"):
        if qa_chain is None:
            st.error("ระบบไม่พร้อมใช้งาน เนื่องจากไม่สามารถโหลดเอกสารอ้างอิงได้")
        else:
            with st.spinner("🔍 กำลังค้นหาข้อมูลในเอกสารและประมวลผล RAG..."):
                start_time = time.time()
                response = qa_chain.invoke({"query": prompt})
                elapsed_time = time.time() - start_time
                
                answer = response.get("result", "")
                sources = response.get("source_documents", [])

                st.markdown(answer)
                st.markdown(f'<div class="time-badge">⏱️ เวลาประมวลผล RAG: {elapsed_time:.2f} วินาที</div>', unsafe_allow_html=True)
                
                if sources:
                    with st.expander(f"🔍 ดูข้อมูลอ้างอิงที่ AI ดึงมาอ่าน ({len(sources)} ชิ้น)"):
                        for i, doc in enumerate(sources):
                            source_file = os.path.basename(doc.metadata.get('source', 'ไม่ทราบไฟล์'))
                            st.markdown(f"**ชิ้นที่ {i+1}** (จากไฟล์: `{source_file}`)")
                            st.code(doc.page_content, language="markdown")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "rag_time": elapsed_time,
                "sources": sources
            })
