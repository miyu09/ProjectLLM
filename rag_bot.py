import os
import sys
import warnings

# 🧹 1. ซ่อนแจ้งเตือนสีเหลืองกวนใจ (DeprecationWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ส่วนของการนำเข้า Library
from langchain_ollama import OllamaLLM, OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma 
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_classic.chains import RetrievalQA
from langchain_core.callbacks.manager import CallbackManager
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ==========================================
# ตั้งค่าระบบ (Config)
# ==========================================
VERSION = "1.1"
MODEL_NAME = "qwen2.5:3b"        # ใช้ Qwen 3B (ความจุระดับกลาง ฉลาดและเร็วพอดีกับ VRAM 4GB)
EMBED_MODEL = "bge-m3"           # โมเดลสมองส่วนจำ (ตัวแปลงภาษา)

def main():
    print(f"--- กำลังเริ่มระบบ RAG บอท (v{VERSION}) ---")
    
    import glob

    # 1. ตรวจสอบและโหลดไฟล์ข้อมูลทั้งหมด
    print("1. กำลังค้นหาและอ่านข้อมูลจากไฟล์เอกสาร (.txt / .md) ทั้งหมด...")
    documents = []
    
    # ค้นหาไฟล์ .txt และ .md ทั้งหมดในโฟลเดอร์ data (ยกเว้น run.txt ถ้าบังเอิญมี)
    data_files = [f for f in glob.glob("data/*.*") if f.endswith(('.txt', '.md')) and not f.endswith("run.txt")]
    
    if not data_files:
        print("ไม่พบไฟล์ข้อมูล (.txt / .md) ในโฟลเดอร์ data เลยครับ")
        return

    for file_path in data_files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
            print(f"   - อ่านไฟล์สำเร็จ: {file_path}")
        except Exception as e:
            print(f"   - ข้ามไฟล์ {file_path} เนื่องจาก Error: {e}")

    # 3. แบ่งข้อมูลเป็นชิ้นๆ (Split Data)
    print("2. กำลังแบ่งข้อมูลเป็นชิ้นย่อย...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) # ลด chunk size ลงเพื่อให้ประมวลผลเร็วขึ้น
    chunks = text_splitter.split_documents(documents)

    # 4. สร้างฐานข้อมูลเวกเตอร์ (Create Vector DB)
    print("3. กำลังสร้างสมองส่วนจำ (Vector DB)... ขั้นตอนนี้อาจนานนิดนึง")
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)

    print(f"4. กำลังเชื่อมต่อกับ {MODEL_NAME}...")

    def get_two_stage_context(vector_store, query, k_grading=2, k_advice=3):
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

    # 🚀 เปิดโหมด Streaming ให้บอทพิมพ์ตอบทีละคำ
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.0,
        repeat_penalty=1.1,
        top_p=0.8,
        num_ctx=8192,
        callbacks=CallbackManager([StreamingStdOutCallbackHandler()])
    ) 
    
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

    print("\n✅ ระบบพร้อมใช้งาน! (พิมพ์ 'exit' เพื่อออก)")
    print("-" * 30)

    # 6. เริ่มวนลูปคุย (Chat Loop)
    while True:
        query = input("\nคุณ: ")
        if query.lower() in ['exit', 'quit', 'ออก']:
            break
        
        if not query.strip():
            continue

        print("🤖 AI ตอบ: ", end="", flush=True)
        try:
            import time
            start_time = time.time()
            
            grading_docs, advice_docs = get_two_stage_context(vector_store, query, k_grading=2, k_advice=3)
            all_sources = grading_docs + advice_docs
            
            context_text = ""
            if grading_docs:
                context_text += "=== [ส่วนที่ 1: เกณฑ์ประเมินเกรด (จากไฟล์ Grading.md)] ===\n"
                for doc in grading_docs:
                    context_text += f"[ไฟล์: {os.path.basename(doc.metadata.get('source',''))}]\n{doc.page_content}\n\n"
                    
            if advice_docs:
                context_text += "=== [ส่วนที่ 2: เนื้อหาคำแนะนำและข้อมูลอ้างอิง (จากไฟล์คำแนะนำ)] ===\n"
                for doc in advice_docs:
                    context_text += f"[ไฟล์: {os.path.basename(doc.metadata.get('source',''))}]\n{doc.page_content}\n\n"

            messages = PROMPT.format_messages(context=context_text, question=query)
            response = llm.invoke(messages)
            elapsed_time = time.time() - start_time
            print("\n")
            print(f"⏱️ เวลาการทำ RAG: {elapsed_time:.2f} วินาที")
            
            print(f"\n[🔍 ข้อมูลอ้างอิงที่ AI ดึงมาอ่าน ({len(all_sources)} ชิ้น: Grading {len(grading_docs)} ชิ้น + คำแนะนำ {len(advice_docs)} ชิ้น)]")
            for i, doc in enumerate(all_sources):
                source_file = doc.metadata.get('source', 'ไม่ทราบไฟล์')
                doc_type = "Grading" if "Grading.md" in source_file else "คำแนะนำ"
                print(f"--- ชิ้นที่ {i+1} [{doc_type}] (จากไฟล์: {os.path.basename(source_file)}) ---")
                print(doc.page_content)
                print("-" * 15)
                
        except Exception as e:
            print(f"\nเกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    main()