import os
import sys
import time
import glob
import csv
import argparse
import warnings
from tqdm import tqdm

# 🧹 ซ่อนแจ้งเตือน Warning ต่างๆ
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ==========================================
# ค่าเริ่มต้นระบบ (Default Config)
# ==========================================
DEFAULT_MODEL = "qwen2.5:1.5b"
DEFAULT_EMBED = "bge-m3"
DEFAULT_INPUT_CSV = "Test.csv"
DEFAULT_K_GRADING = 2
DEFAULT_K_ADVICE = 3
CHROMA_PATH = "./chroma_db"

def load_or_create_vectorstore(embed_model_name):
    """โหลด หรือสร้าง Vector DB จากโฟลเดอร์ data"""
    embeddings = OllamaEmbeddings(model=embed_model_name)
    
    if os.path.exists(CHROMA_PATH) and os.listdir(CHROMA_PATH):
        print(f"📦 กำลังโหลด Vector Store จาก: {CHROMA_PATH}")
        return Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    print("🔨 ไม่พบ Chroma DB เดิม กำลังอ่านไฟล์ใน data/ เพื่อสร้างใหม่...")
    data_files = [f for f in glob.glob("data/*.*") if f.endswith(('.txt', '.md')) and not f.endswith("run.txt")]
    if not data_files:
        raise FileNotFoundError("ไม่พบไฟล์เอกสารในโฟลเดอร์ data/")
        
    documents = []
    for file_path in data_files:
        try:
            loader = TextLoader(file_path, encoding='utf-8')
            documents.extend(loader.load())
        except Exception as e:
            print(f"⚠️ ข้ามไฟล์ {file_path}: {e}")
            
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)
    vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=CHROMA_PATH)
    print("✅ สร้าง Vector Store สำเร็จ!")
    return vector_store

def get_two_stage_context(vector_store, query, k_grading=2, k_advice=3):
    """ดึงข้อมูล 2 Stage: Grading (เกณฑ์) และ Advice (คำแนะนำ)"""
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

def parse_input_csv(csv_path):
    """อ่านคำถามและแยกตามหมวดหมู่จากไฟล์ CSV"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ไม่พบไฟล์คำถาม: {csv_path}")
        
    categories = []
    current_category = "ทั่วไป"
    questions_data = []

    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or not any(row):
                continue
            
            first_cell = row[0].strip()
            
            # ตรวจจับหัวหมวดหมู่ (เช่น อาการท้องผูก, อาการระบบประสาท, อาการแผลในปาก)
            if first_cell.startswith("อาการ"):
                current_category = first_cell
                continue
                
            # ข้ามแถว Header ของตาราง
            if first_cell in ["ลำดับ", "Gwen 2.5 (7B) (k=5)"] or first_cell.startswith("Gwen") or first_cell.startswith("Qwen"):
                continue
                
            # ตรวจว่าเป็นแถวข้อมูลคำถาม (คอลัมน์ 0 = ลำดับ, คอลัมน์ 1 = คำถาม)
            if first_cell.isdigit() and len(row) >= 2:
                q_id = first_cell
                q_text = row[1].strip()
                if q_text:
                    questions_data.append({
                        "category": current_category,
                        "id": q_id,
                        "question": q_text
                    })
                    
    return questions_data

def build_prompt_template():
    """สร้าง System & Human Prompt Template"""
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
   - Do NOT invent strange, absurd, or unscientific actions.
   - Provide ONLY standard, medically safe self-care advice found in the reference context.
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

    return ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])

def run_batch_eval(model_name, embed_model, input_csv, output_csv, k_grading, k_advice):
    """รัน Batch Test สำหรับชุดคำถามทั้งหมด"""
    total_k = k_grading + k_advice
    print(f"\n==========================================")
    print(f"🚀 เริ่มต้นการทดสอบ Batch RAG Evaluation")
    print(f"🤖 LLM Model   : {model_name}")
    print(f"🧠 Embed Model : {embed_model}")
    print(f"🎯 Search K    : Total K={total_k} (Grading={k_grading}, Advice={k_advice})")
    print(f"📄 Input File  : {input_csv}")
    print(f"💾 Output File : {output_csv}")
    print(f"==========================================\n")

    # 1. โหลดข้อมูลคำถาม
    questions = parse_input_csv(input_csv)
    print(f"📋 โหลดคำถามสำเร็จทั้งหมด {len(questions)} ข้อ\n")

    # 2. โหลด Vector Store & LLM
    vector_store = load_or_create_vectorstore(embed_model)
    
    llm = ChatOllama(
        model=model_name,
        temperature=0.0,
        repeat_penalty=1.1,
        top_p=0.8,
        num_ctx=8192,
        num_predict=400  # 🛑 จำกัดการสร้างข้อความสูงสุด 400 tokens ป้องกันโมเดลวนลูปค้าง
    )
    
    prompt_template = build_prompt_template()

    # 3. เตรียมไฟล์ CSV และเขียน Header
    fieldnames = [
        "Model",
        "K_Config",
        "หมวดหมู่",
        "ลำดับ",
        "คำถาม",
        "คำตอบ",
        "ตอบถูกหรือผิด",
        "chunk ถูกหรือผิด",
        "เวลา (วินาที)",
        "Vector Sources (ไฟล์อ้างอิง)",
        "Vector Chunks Preview"
    ]

    with open(output_csv, mode='w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        # 4. วนลูปทดสอบแต่ละคำถาม
        for item in tqdm(questions, desc="Processing Questions", unit="q"):
            category = item["category"]
            q_id = item["id"]
            question_text = item["question"]

            start_time = time.time()
            
            try:
                # ดึง Context
                grading_docs, advice_docs = get_two_stage_context(vector_store, question_text, k_grading=k_grading, k_advice=k_advice)
                all_sources = grading_docs + advice_docs
                
                # รวม Text ของ Context
                context_text = ""
                if grading_docs:
                    context_text += "=== [ส่วนที่ 1: เกณฑ์ประเมินเกรด (จากไฟล์ Grading.md)] ===\n"
                    for doc in grading_docs:
                        context_text += f"[ไฟล์: {os.path.basename(doc.metadata.get('source',''))}]\n{doc.page_content}\n\n"
                        
                if advice_docs:
                    context_text += "=== [ส่วนที่ 2: เนื้อหาคำแนะนำและข้อมูลอ้างอิง (จากไฟล์คำแนะนำ)] ===\n"
                    for doc in advice_docs:
                        context_text += f"[ไฟล์: {os.path.basename(doc.metadata.get('source',''))}]\n{doc.page_content}\n\n"

                # เรียก LLM
                messages = prompt_template.format_messages(context=context_text, question=question_text)
                response = llm.invoke(messages)
                elapsed_time = time.time() - start_time
                answer_text = response.content.strip()

                # สรุปรายการไฟล์ Vector Sources และเรียง Chunk ขึ้นบรรทัดใหม่ทีละชิ้น
                source_files_summary = []
                sources_detail_list = []
                for idx, doc in enumerate(all_sources, start=1):
                    src_name = os.path.basename(doc.metadata.get('source', 'ไม่ระบุ'))
                    source_files_summary.append(src_name)
                    # ทำความสะอาด newline ภายใน chunk เพื่อให้ 1 chunk อยู่ใน 1 บรรทัดของเซลล์
                    clean_chunk = doc.page_content.replace('\r\n', ' ').replace('\n', ' ').strip()
                    sources_detail_list.append(f"[Chunk {idx}] [{src_name}]: {clean_chunk}")
                    
                sources_summary_str = ", ".join(list(dict.fromkeys(source_files_summary)))
                sources_chunks_preview = "\n\n".join(sources_detail_list)

            except Exception as e:
                elapsed_time = time.time() - start_time
                answer_text = f"ERROR: เกิดข้อผิดพลาดในการประมวลผล - {e}"
                sources_summary_str = "N/A"
                sources_chunks_preview = str(e)

            # บันทึกลงไฟล์ CSV ทันที
            row_data = {
                "Model": model_name,
                "K_Config": f"k={total_k} (Grading={k_grading}, Advice={k_advice})",
                "หมวดหมู่": category,
                "ลำดับ": q_id,
                "คำถาม": question_text,
                "คำตอบ": answer_text,
                "ตอบถูกหรือผิด": "",
                "chunk ถูกหรือผิด": "FALSE",
                "เวลา (วินาที)": f"{elapsed_time:.2f}",
                "Vector Sources (ไฟล์อ้างอิง)": sources_summary_str,
                "Vector Chunks Preview": sources_chunks_preview
            }
            writer.writerow(row_data)
            f.flush()

    print(f"\n🎉 รันทดสอบครบ {len(questions)} ข้อ สำเร็จแล้ว!")
    print(f"📁 บันทึกผลลัพธ์ไว้ที่: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Automated RAG Batch Evaluation Tool")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help=f"ชื่อ LLM Model ใน Ollama (default: {DEFAULT_MODEL})")
    parser.add_argument("--embed", type=str, default=DEFAULT_EMBED, help=f"ชื่อ Embedding Model (default: {DEFAULT_EMBED})")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_CSV, help=f"ไฟล์ CSV ตั้งต้นที่มีคำถาม (default: {DEFAULT_INPUT_CSV})")
    parser.add_argument("--output", type=str, default="", help="ชื่อไฟล์ CSV ผลลัพธ์ที่จะบันทึก (หากไม่ระบุจะสร้างชื่อให้อัตโนมัติ)")
    parser.add_argument("--k_grading", type=int, default=DEFAULT_K_GRADING, help=f"จำนวน K สำหรับ Grading docs (default: {DEFAULT_K_GRADING})")
    parser.add_argument("--k_advice", type=int, default=DEFAULT_K_ADVICE, help=f"จำนวน K สำหรับ Advice docs (default: {DEFAULT_K_ADVICE})")

    args = parser.parse_args()

    # ตั้งชื่อไฟล์ output อัตโนมัติถ้าไม่ได้ระบุ
    if not args.output:
        clean_model_name = args.model.replace(":", "_").replace("/", "_")
        total_k = args.k_grading + args.k_advice
        args.output = f"results_{clean_model_name}_k{total_k}.csv"

    run_batch_eval(
        model_name=args.model,
        embed_model=args.embed,
        input_csv=args.input,
        output_csv=args.output,
        k_grading=args.k_grading,
        k_advice=args.k_advice
    )

if __name__ == "__main__":
    main()
