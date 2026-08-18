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

MODEL_NAME = "qwen2.5:3b"        # ใช้ Qwen 3B (ความจุระดับกลาง ฉลาดและเร็วพอดีกับ VRAM 4GB)
EMBED_MODEL = "bge-m3"           # โมเดลสมองส่วนจำ (ตัวแปลงภาษา)

def main():
    print(f"--- กำลังเริ่มระบบ RAG บอท ---")
    
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

    # 🚀 เปิดโหมด Streaming ให้บอทพิมพ์ตอบทีละคำ
    llm = ChatOllama(
        model=MODEL_NAME,
        temperature=0.0,
        num_ctx=8192,          # เพิ่มบริบทให้กว้างขึ้น เพื่อป้องกันปัญหาคำสั่งถูกตัดเมื่อดึงข้อมูลหลายชิ้น (k=5)
        callbacks=CallbackManager([StreamingStdOutCallbackHandler()])
    ) 
    
    # โหลดเกณฑ์การประเมิน (Grading) เข้าสู่ System Prompt ถาวร
    grading_criteria = ""
    try:
        with open("data/Grading.md", "r", encoding="utf-8") as f:
            grading_criteria = f.read()
    except Exception as e:
        grading_criteria = "ไม่พบไฟล์ Grading.md"
        
    system_template = f"""คุณคือผู้ช่วย AI ที่มีหน้าที่ประเมินอาการและตอบคำถามจากข้อมูลอ้างอิงที่กำหนดให้เท่านั้น
ข้อบังคับสำคัญที่สุด:
1. คุณต้องตอบเป็น "ภาษาไทย" เท่านั้น (Strictly answer in Thai language). ห้ามมีตัวอักษรภาษาจีนเด็ดขาด (NO CHINESE).
2. ให้ประเมิน Grade โดยอ้างอิงจาก "ข้อมูลอ้างอิง" ด้านล่างนี้เป็นหลัก
3. สำหรับส่วน "คำแนะนำเบื้องต้นสำหรับคนไข้" อนุญาตให้ใช้ความรู้ทางการแพทย์ของคุณเองในการให้คำแนะนำที่ปลอดภัยและเหมาะสมได้
4. บังคับรูปแบบการตอบให้ใช้ Format นีัเท่านั้น ห้ามมีเลขข้อย่อย:

ประเมินระดับความรุนแรง:
(คุณต้อง "ประเมินและเลือก" Grade ที่ตรงกับอาการของผู้ป่วยมากที่สุด "เพียง 1 ระดับเท่านั้น". หากอาการไม่ตรงแบบเป๊ะๆ อนุโลมให้คุณประเมินเทียบเคียงเกรดที่ใกล้เคียงที่สุดได้. ห้ามคัดลอก Grade อื่นๆ มาแสดง พร้อมอธิบายเหตุผลสั้นๆ)

คำแนะนำเบื้องต้นสำหรับคนไข้:
(ให้คำแนะนำสำหรับคนไข้)
4. หากพบอาการอันตรายในข้อมูลให้เน้นย้ำผู้ใช้ด้วย

=== GRADING CRITERIA ===
{grading_criteria}
======================="""
    
    human_template = """ข้อมูลอ้างอิง:
{context}

คำถาม: {question}"""

    PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template(human_template)
    ])
    
    # สร้างรูปแบบการใส่ข้อมูลอ้างอิงให้ AI เห็นไฟล์ต้นทาง
    document_prompt = PromptTemplate(
        input_variables=["page_content", "source"],
        template="[ไฟล์: {source}]\n{page_content}"
    )

    # สร้างระบบถาม-ตอบ พร้อมฝัง Prompt ที่เราตั้งไว้
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}), 
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PROMPT,
            "document_prompt": document_prompt
        } # 🔒 บังคับใช้ Prompt
    )

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
            # ส่งคำถามไปให้ระบบ RAG พร้อมจับเวลา
            import time
            start_time = time.time()
            response = qa_chain.invoke({"query": query})
            elapsed_time = time.time() - start_time
            print("\n") # เว้นบรรทัดเมื่อตอบจบประโยค
            print(f"⏱️ เวลาการทำ RAG: {elapsed_time:.2f} วินาที")
            
            # --- ส่วนสำหรับแอบดูสมอง AI (แสดง Chunk ที่ถูกดึงมา) ---
            print(f"\n[🔍 ข้อมูลอ้างอิงที่ AI ดึงมาอ่าน ({len(response['source_documents'])} ชิ้น)]")
            for i, doc in enumerate(response['source_documents']):
                source_file = doc.metadata.get('source', 'ไม่ทราบไฟล์')
                print(f"--- ชิ้นที่ {i+1} (จากไฟล์: {os.path.basename(source_file)}) ---")
                print(doc.page_content)
                print("-" * 15)
                
        except Exception as e:
            print(f"\nเกิดข้อผิดพลาด: {e}")

if __name__ == "__main__":
    main()