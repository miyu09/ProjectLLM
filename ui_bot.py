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
VERSION = "1.1"
MODEL_NAME = "qwen2.5:3b"        # ใช้ Qwen 3B (ความจุระดับกลาง ฉลาดและเร็วพอดีกับ VRAM 4GB)
EMBED_MODEL = "bge-m3"           # โมเดลสมองส่วนจำ (ตัวแปลงภาษา)

# Page Config
st.set_page_config(
    page_title=f"Medical Assessment RAG Bot v{VERSION}",
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
def load_rag_components():
    """โหลด Vector DB และ LLM พร้อมแคชไว้ในหน่วยความจำเพื่อความรวดเร็ว"""
    data_files = [f for f in glob.glob("data/*.*") if f.endswith(('.txt', '.md')) and not f.endswith("run.txt")]
    if not data_files:
        return None, None, "ไม่พบไฟล์ข้อมูล (.txt / .md) ในโฟลเดอร์ data"

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
        return None, None, "ไม่สามารถอ่านไฟล์ในโฟลเดอร์ data ได้"

    CHROMA_PATH = "./chroma_db"
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    else:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)

    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.0,
        repeat_penalty=1.1,
        top_p=0.8,
        num_ctx=8192
    )

    return vector_store, llm, loaded_files

def get_two_stage_context(vector_store, query, k_grading=2, k_advice=3):
    """
    ดึงข้อมูล 2 ส่วนอย่างเด็ดขาด:
    1. grading_docs: ดึง K ชิ้นเฉพาะจากไฟล์ Grading.md เท่านั้น
    2. advice_docs: ดึง K ชิ้นจากไฟล์อื่นๆ ทั้งหมด (ยกเว้น Grading.md)
    """
    all_docs = vector_store.similarity_search(query, k=25)
    
    grading_docs = []
    advice_docs = []
    
    for doc in all_docs:
        src = os.path.basename(doc.metadata.get("source", ""))
        if "Grading.md" in src:
            if len(grading_docs) < k_grading:
                grading_docs.append(doc)
        else:
            if len(advice_docs) < k_advice:
                advice_docs.append(doc)
                
    return grading_docs, advice_docs

# --- Header ---
st.markdown(f'<div class="main-header">🏥 Medical Assessment RAG Bot <span style="font-size: 1.2rem; color: #64748B;">v{VERSION}</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">ระบบประเมินอาการและตอบคำถามทางการแพทย์ด้วย RAG (ประเมินเกรด + คำแนะนำที่ปลอดภัย)</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header(f"⚙️ ตั้งค่า RAG & ระบบ (v{VERSION})")
    st.write(f"🤖 **LLM Model:** `{MODEL_NAME}`")
    st.write(f"🧠 **Embedding Model:** `{EMBED_MODEL}`")
    
    st.divider()
    
    st.subheader("🎯 ตั้งค่าการดึงข้อมูล (K-Search)")
    k_grading = st.slider("K (Grading.md)", min_value=1, max_value=5, value=2, help="จำนวน Chunk จากไฟล์เกณฑ์ประเมินเกรด")
    k_advice = st.slider("K (ไฟล์คำแนะนำอื่นๆ)", min_value=1, max_value=5, value=3, help="จำนวน Chunk จากไฟล์เนื้อหาคำแนะนำ")

    st.divider()
    
    vector_store, llm, loaded_files = load_rag_components()
    
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

# --- System Prompt Construction ---
system_template = """You are an expert oncology medical AI assistant for doctors and chemotherapy patients.
Your sole responsibility is to evaluate symptom severity grades and provide safe, medical-grade patient advice strictly based on the provided reference context.

CRITICAL MEDICAL SAFETY & GUARDRAIL RULES:
1. STRICT THAI LANGUAGE ONLY (ห้ามมีภาษาจีนเด็ดขาด):
   - You MUST answer ENTIRELY in natural, polite Thai language.
   - Absolutely ZERO Chinese characters or Chinese vocabulary allowed under any circumstances.
2. NO SELF-MEDICATION ADVICE (ห้ามแนะนำให้คนไข้ไปซื้อยาทานเองเด็ดขาด):
   - NEVER suggest, recommend, or instruct patients to buy or take over-the-counter medicines on their own.
   - ALWAYS advise the patient to consult their oncology doctor, attending physician, or healthcare team before taking any medicine or modifying treatment.
3. NO BIZARRE OR UNREALISTIC ADVICE (ห้ามคิดค้นวิธีปฏิบัติที่ไม่สมจริง):
   - Do NOT invent strange, absurd, or unscientific actions (such as using feet instead of hands, or unrealistic body movements).
   - Provide ONLY standard, medically safe self-care advice found in the reference context (such as drinking warm water, resting, avoiding spicy/rough foods, observing symptoms, or contacting their hospital).
4. ACCURATE CTCAE GRADING:
   - Compare reported symptoms against the provided "เกณฑ์ประเมินเกรด (Grading Criteria)".
   - Select ONLY ONE exact Grade (Grade 1, Grade 2, Grade 3, Grade 4, or Grade 5) that best matches the patient's condition, with a concise Thai explanation.

REQUIRED RESPONSE FORMAT (ต้องตอบตามรูปแบบนี้เท่านั้น):

ประเมินระดับความรุนแรง:
Grade X (คำอธิบายสั้นๆ เกี่ยวกับอาการที่ตรงกับเกรด และเหตุผลที่ประเมินเป็นเกรดนี้)

คำแนะนำเบื้องต้นสำหรับคนไข้:
(คำแนะนำการดูแลตัวเองที่ปลอดภัยตามข้อมูลอ้างอิง อธิบายสาเหตุเบื้องต้น และเน้นย้ำให้ขอคำแนะนำจากแพทย์ประจำตัวเสมอ ห้ามแนะนำให้ไปซื้อยาทานเองเด็ดขาด)

STRICT LANGUAGE REMINDER: Respond ONLY in Thai. NO Chinese characters."""

human_template = """Reference Context:
{context}

Patient Query: {question}"""

PROMPT = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(human_template)
])

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
        if vector_store is None or llm is None:
            st.error("ระบบไม่พร้อมใช้งาน เนื่องจากไม่สามารถโหลดเอกสารอ้างอิงได้")
        else:
            with st.spinner("🔍 กำลังค้นหาข้อมูลแยก 2 ส่วน (Grading + คำแนะนำ) และประมวลผล RAG..."):
                start_time = time.time()
                
                # 1. ดึงข้อมูล 2 ส่วนอย่างเด็ดขาด
                grading_docs, advice_docs = get_two_stage_context(vector_store, prompt, k_grading=k_grading, k_advice=k_advice)
                all_sources = grading_docs + advice_docs
                
                # 2. จัดรูปแบบ Context แยกหมวดหมู่ให้ AI อ่าน
                context_text = ""
                if grading_docs:
                    context_text += "=== [ส่วนที่ 1: เกณฑ์ประเมินเกรด (จากไฟล์ Grading.md)] ===\n"
                    for doc in grading_docs:
                        context_text += f"[ไฟล์: {os.path.basename(doc.metadata.get('source',''))}]\n{doc.page_content}\n\n"
                        
                if advice_docs:
                    context_text += "=== [ส่วนที่ 2: เนื้อหาคำแนะนำและข้อมูลอ้างอิง (จากไฟล์คำแนะนำ)] ===\n"
                    for doc in advice_docs:
                        context_text += f"[ไฟล์: {os.path.basename(doc.metadata.get('source',''))}]\n{doc.page_content}\n\n"

                # 3. ส่งคำสั่งให้ LLM ประมวลผล
                messages = PROMPT.format_messages(context=context_text, question=prompt)
                response = llm.invoke(messages)
                elapsed_time = time.time() - start_time
                
                answer = response.content if hasattr(response, 'content') else str(response)

                st.markdown(answer)
                st.markdown(f'<div class="time-badge">⏱️ เวลาประมวลผล RAG: {elapsed_time:.2f} วินาที</div>', unsafe_allow_html=True)
                
                if all_sources:
                    with st.expander(f"🔍 ดูข้อมูลอ้างอิงที่ AI ดึงมาอ่าน ({len(all_sources)} ชิ้น: Grading {len(grading_docs)} ชิ้น + คำแนะนำ {len(advice_docs)} ชิ้น)"):
                        for i, doc in enumerate(all_sources):
                            source_file = os.path.basename(doc.metadata.get('source', 'ไม่ทราบไฟล์'))
                            doc_type = "Grading" if "Grading.md" in source_file else "คำแนะนำ"
                            st.markdown(f"**ชิ้นที่ {i+1} [{doc_type}]** (จากไฟล์: `{source_file}`)")
                            st.code(doc.page_content, language="markdown")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "rag_time": elapsed_time,
                "sources": all_sources
            })

