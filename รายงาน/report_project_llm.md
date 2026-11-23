# การพัฒนาระบบสถาปัตยกรรมหลังบ้านแชทบอทประเมินระดับความรุนแรงของผลข้างเคียงสำหรับผู้ป่วยเคมีบำบัดด้วยเทคโนโลยี Retrieval-Augmented Generation

### A DEVELOPMENT OF BACKEND RETRIEVAL-AUGMENTED GENERATION (RAG) ARCHITECTURE FOR CHEMOTHERAPY SYMPTOM EVALUATION AND INTEGRATION WITH NARESUAN UNIVERSITY HOSPITAL CHATBOT SYSTEM

---

## บทคัดย่อ (Abstract - Thai)

โครงงานนี้นำเสนอการวิจัยและพัฒนาสถาปัตยกรรมระบบหลังบ้านอัจฉริยะ (Backend Intelligent Architecture) โดยใช้เทคโนโลยี Retrieval-Augmented Generation (RAG) สำหรับตอบคำถามและประเมินระดับความรุนแรงของผลข้างเคียงจากยาเคมีบำบัด เพื่อเตรียมเชื่อมต่อเข้ากับระบบแชทบอทเดิมของโรงพยาบาลมหาวิทยาลัยนเรศวร โดยมุ่งเน้นการคัดกรองภาวะแทรกซ้อน 3 กลุ่มอาการหลักที่พบบ่อยในผู้ป่วยโรคมะเร็ง ได้แก่ แผลในช่องปาก (Oral Mucositis), ท้องผูก (Chemotherapy-Induced Constipation), และปลายประสาทอักเสบหรือชาปลายมือปลายเท้า (Peripheral Neuropathy) ระบบหลังบ้านได้รับการออกแบบให้เชื่อมโยงฐานข้อมูลเวกเตอร์แบบฝังตัว (Embedded Vector Store) และโมเดลแปลงเวกเตอร์หลายภาษา (Multilingual Dense Embedding) เข้ากับคลังความรู้ทางการแพทย์และเกณฑ์มาตรฐานสากล Common Terminology Criteria for Adverse Events (CTCAE v5.0)

ในขั้นตอนการวิจัยและพัฒนา ได้มีการสร้างส่วนต่อประสานโปรแกรมประยุกต์ต้นแบบด้วย Streamlit ขึ้นเพื่อใช้เป็นเครื่องมือทดสอบสมรรถนะภายใน (Testing & Evaluation Prototype) โดยดำเนินการทดสอบเปรียบเทียบระหว่างโมเดลภาษาขนาดเล็ก (Small Language Models: SLM) ทั้งหมด 5 โมเดล บนชุดคำถามเคสจำลองทางการแพทย์โมเดลละ 30 คำถาม เพื่อวัดความถูกต้องในการจัดระดับความรุนแรงของอาการ ความแม่นยำในการค้นคืนบริบททางการแพทย์ และความเร็วในการตอบสนอง (Latency) นอกจากนี้ ระบบหลังบ้านได้ถูกวางโครงสร้างแบบแยกส่วน (Decoupled Headless Backend) พร้อมกลไกควบคุมความปลอดภัยทางการแพทย์ (Medical Guardrails) และ Application Programming Interface (API) เพื่อเตรียมความพร้อมสำหรับการส่งมอบและเชื่อมต่อเข้ากับแพลตฟอร์มบริการสนทนาเดิมของโรงพยาบาลมหาวิทยาลัยนเรศวรในสภาพแวดล้อม On-Premise อย่างปลอดภัยและรักษาความเป็นส่วนตัวของข้อมูลผู้ป่วย

---

## Abstract (English)

This project presents the design, architectural formulation, and empirical evaluation of a specialized Backend Retrieval-Augmented Generation (RAG) engine engineered for chemotherapy symptom severity assessment and patient advisory. The backend service is purposefully designed to integrate via Application Programming Interfaces (APIs) with the pre-existing hospital conversational chatbot platform at Naresuan University Hospital. The clinical scope concentrates on three high-prevalence chemotherapy complications: oral mucositis, chemotherapy-induced constipation, and peripheral sensory neuropathy. By decoupling linguistic generation from medical parametric memory, the backend architecture integrates high-density multilingual embeddings and an embedded vector store to dynamically retrieve evidence from curated clinical knowledge bases, synthesizing standardized Common Terminology Criteria for Adverse Events (CTCAE v5.0) criteria with evidence-based supportive care protocols.

During the development and validation phase, a lightweight Streamlit web interface was deployed strictly as an internal testing testbed and experimental evaluation prototype. Benchmark evaluations were executed across five language model candidate architectures using a standardized protocol of 30 clinical case queries per model to assess severity grading accuracy, medical context retrieval recall, and inference latency under local and high-performance Cloud GPU environments. The backend architecture enforces strict clinical guardrails, red-flag escalation logic, and a modular service layer, establishing a production-ready foundation for on-premise deployment and seamless interoperability with Naresuan University Hospital's existing digital health infrastructure.

---

# Chapter 1: Introduction

### 1.1 Background and Significance of the Project
Chemotherapy remains a cornerstone of systemic therapeutic regimens in modern clinical oncology. However, because cytotoxic agents non-selectively impair cellular proliferation throughout the human body, patients routinely suffer from severe adverse toxicities. Among the most prevalent and debilitating conditions reported during outpatient convalescence are oral mucositis (mucosal inflammation, severe pain, and deep ulceration), chemotherapy-induced constipation, and peripheral sensory neuropathy (stocking-and-glove sensory loss, tingling, and tactile impairment). If left unmonitored and improperly managed at home, these toxicities rapidly undermine patient nutrition, compromise physical autonomy, and induce severe psychological distress, frequently culminating in unplanned emergency department visits, chemotherapy dose reductions, or fatal treatment interruptions.

Between outpatient clinical visits, cancer patients face a substantial "semantic and communicative barrier." While healthcare institutions—such as Naresuan University Hospital—provide comprehensive patient information sheets and reference standardized clinical guidelines like the Common Terminology Criteria for Adverse Events (CTCAE), patients and family caregivers often struggle to decipher complex medical classifications. Instead, patients communicate their symptoms in colloquial, subjective terms (e.g., describing severe mouth ulcers as an inability to swallow fluids, or advanced neuropathy as frequently dropping utensils while cooking). Conventional keyword-matching algorithms and generic internet search engines fail to comprehend the clinical significance of these informal complaints, frequently returning fragmented or misleading advice. Simultaneously, hospital telephone advice lines face significant staffing constraints outside regular operating hours, creating an acute demand for intelligent, around-the-clock clinical guidance.

To enhance outpatient communication and clinical service accessibility, Naresuan University Hospital maintains an established hospital chatbot platform. However, traditional rule-based or standard conversational chatbots lack the clinical reasoning required to accurately interpret unstructured medical narratives, assess complication severity grades, and synthesize multi-faceted nursing advice. While Large Language Models (LLMs) offer advanced linguistic understanding, standalone generative models suffer from "hallucinations"—generating confident yet medically fabricated facts, incorrect dosages, or contraindicated remedies. In high-stakes oncology care, ungrounded generative AI introduces unacceptable liability and safety risks.

To overcome these challenges, this project focuses on designing and developing a specialized **Backend Medical Assessment RAG System**. Operating on localized Small Language Models (SLMs), dense multilingual semantic embeddings, and an embedded vector store, the backend pipeline implements a segregated dual-stream retrieval strategy that concurrently retrieves standardized CTCAE clinical grading thresholds and supportive nursing care guidelines. To facilitate rigorous testing and parameter optimization, an agile Streamlit interface was implemented strictly as an internal experimental testbed. The ultimate objective is to provide a robust, API-driven backend intelligence engine that connects directly with Naresuan University Hospital’s existing chatbot infrastructure, empowering cancer patients with 24/7 evidence-based symptom triage while upholding data privacy and clinical safety.

---

### 1.2 Project Objectives
1. **To design and architect an intelligent Backend Medical Assessment RAG Engine** capable of translating unstructured, colloquial patient symptom descriptions into standardized CTCAE severity grades (Grade 1–4) and safe, evidence-based home-care guidance.
2. **To implement a multi-source dual-stream retrieval mechanism** within the backend pipeline that systematically segregates clinical threshold criteria from supportive nursing care protocols, ensuring balanced context synthesis.
3. **To evaluate and compare model performance, severity grading accuracy, and inference latency** across five language model candidates using a standardized benchmark of 30 clinical case questions per model under local and Cloud GPU computing environments.
4. **To formulate strict medical guardrails and safety protocols** that eliminate hallucinations, restrict unprescribed pharmacological recommendations, and automatically triage Grade 3–4 high-risk symptoms for urgent clinical referral.
5. **To deploy an internal Streamlit testbed prototype** for experimental validation and latency instrumentation, establishing standard API interfaces for seamless integration into Naresuan University Hospital's existing chatbot platform.

---

### 1.3 Expected Benefits

#### 1.3.1 Patient Well-being and Navigational Relief
* **24/7 Context-Aware Symptom Evaluation:** Delivers immediate, empathetic, and evidence-grounded guidance for chemotherapy complications at any time, substantially easing patient anxiety between hospital appointments.
* **Objective Symptom Severity Tracking:** Enables patients and caregivers to accurately gauge complication severity using standardized CTCAE clinical grades explained in clear layman terms.
* **Safer Home Recovery:** Educates patients on safe oral hygiene, dietary adjustments, hydration schedules, and fall-prevention measures, reinforcing therapy compliance.

#### 1.3.2 Healthcare System and Clinical Workflow Optimization
* **Seamless Integration with Hospital Services:** Enhances Naresuan University Hospital’s existing patient chatbot with specialized oncology intelligence without requiring patients to install new external applications.
* **Clinical Triage Support and Early Escalation:** Automatically identifies Grade 3–4 life-threatening complications (e.g., severe obstipation, complete inability to swallow liquids, profound motor impairment) and urges immediate physician contact.
* **Mitigation of Routine Inquiries:** Alleviates repetitive telephone advisory inquiries on outpatient oncology nursing staff, allowing specialized clinicians to prioritize acute in-hospital care.

#### 1.3.3 Technological and Institutional Innovations
* **Privacy-Preserving On-Premise AI Blueprint:** Establishes a functional localized inference architecture operating within the hospital's internal infrastructure, ensuring strict compliance with health data privacy regulations (HIPAA/PDPA) without external cloud data leakage.
* **Modular Headless Backend Design:** Provides an academic reference architecture demonstrating how an intelligent RAG backend can serve as an AI middleware layer for legacy institutional conversational systems.

---

### 1.4 Project Scope

#### 1.4.1 Clinical Symptom Domain and Taxonomy
The project restricts its clinical knowledge base and assessment scope to three critical chemotherapy-induced adverse effects:
1. **Oral Mucositis / Mouth Ulcers** (*MedDRA Code: 10028130*): Covers mucosal redness, painful ulcerations, swallowing difficulty, and nutritional adjustments.
2. **Chemotherapy-Induced Constipation** (*MedDRA Code: 10010774*): Covers defecation difficulties, stool impaction, abdominal distention, dietary fiber balance, and hydration management.
3. **Peripheral Sensory Neuropathy** (*MedDRA Code: 10034620*): Covers tingling sensations, glove-and-stocking numbness, tactile loss, and fine motor impairment (e.g., buttoning clothes, handling utensils).
* **Severity Grading Standard:** Grounded strictly in the National Cancer Institute's Common Terminology Criteria for Adverse Events (CTCAE v5.0), classifying conditions into Grade 1 (Mild), Grade 2 (Moderate), Grade 3 (Severe), and Grade 4 (Life-threatening).

#### 1.4.2 System Architecture and Technical Stack
* **Core Contribution:** Design and implementation of the **Backend RAG Processing Engine and Service Layer**.
* **Language Models:** Evaluated across five open-weight model architectures (spanning Small and Medium Language Model parameter scales) served via an optimized model runtime engine.
* **Embedding Model:** Multilingual dense embedding model supporting dense semantic text representations in both Thai and English.
* **Vector Database:** Lightweight embedded vector database featuring persistent on-disk storage.
* **Orchestration Framework:** Comprehensive LLM orchestration framework managing document chunking, prompt templates, and conversational context pipelines.
* **Testing & Prototyping Interface:** A lightweight Streamlit web application developed strictly as an internal testing environment (Testbed UI) for benchmarking, parameter tuning, and latency monitoring.
* **Target Production Environment:** API-level integration with the existing conversational chatbot system of Naresuan University Hospital on hospital internal on-premise infrastructure.

#### 1.4.3 Evaluation Scope
* **Benchmark Protocol:** Evaluated across 5 language model architectures using a standardized set of 30 clinical case questions per model.
* **Evaluation Metrics:** Severity Grading Accuracy (Correct vs. Incorrect), Context Chunk Retrieval Precision, Response Latency (seconds), and Qualitative Adherence to Medical Safety Guardrails.

---

### 1.5 Methodology
The research and development process was conducted across eight systematically sequenced phases:
1. **Clinical Requirement & Knowledge Elicitation:** Formulating clinical symptom taxonomies and collecting colloquial patient inquiries, slang, and common outpatient questions in coordination with oncology care guidelines.
2. **Medical Knowledge Base Construction:** Compiling, verifying, and structuring evidence-based oncology nursing manuals and clinical grading criteria into standardized textual knowledge repositories.
3. **Document Ingestion and Vector Indexing:** Performing recursive text chunking with semantic overlap and generating high-dimensional dense embeddings stored within the embedded vector database.
4. **Dual-Stream Retrieval Pipeline Engineering:** Implementing custom retrieval filtering that simultaneously fetches clinical threshold criteria chunks and supportive care advice chunks.
5. **Prompt Engineering and Medical Guardrail Hardening:** Designing structured system prompts enforcing clinical role boundaries, mandatory CTCAE grading, refusal of unauthorized prescriptions, and emergency escalation.
6. **Internal Testbed Prototype Development:** Building a Streamlit application used exclusively as an internal testbed to evaluate latency, test prompt variations, and configure retrieval depth parameters ($k$-values).
7. **Empirical Batch Evaluation:** Executing systematic automated benchmarks across five language models using 30 clinical case questions per model to analyze accuracy, retrieval recall, and inference latency.
8. **API Architecture and Hospital Integration Planning:** Designing standard RESTful API endpoints and data exchange contracts for integrating the validated backend engine into Naresuan University Hospital’s existing chatbot.

---

### 1.6 Operational Plan
The project lifecycle is structured across five distinct operational phases over a five-month timeline:

| Phase | Duration / Month | Milestone & Core Activities | Infrastructure & Target |
| :--- | :---: | :--- | :--- |
| **Phase 1** | October (ตุลาคม) | **Local Model & RAG Prototype:** Test initial lightweight models on local hardware, construct the CTCAE knowledge repository, and build the Streamlit testing prototype. | Local Workstation (4GB VRAM GPU / CPU) ($0) |
| **Phase 2** | November (พฤศจิกายน) | **Cloud GPU Testing (Medium & Large Models):** Deploy high-performance Cloud GPU instances, benchmark 30 clinical questions per model across candidates, tune prompt structures and retrieval depths. | Cloud GPU (NVIDIA RTX 4090 24GB VRAM) ($37.00 ~ 1,300 THB for 50 hrs) |
| **Phase 3** | December (ธันวาคม) | **Hospital Server Deployment & Integration:** Freeze the optimal backend model, deploy to Naresuan University Hospital internal server (On-premise), connect API to hospital's existing chatbot, and conduct User Acceptance Testing (UAT) with clinical doctors. | Hospital Server Infrastructure ($0) |
| **Phase 4** | January (มกราคม) | **System Polishing & Final Documentation:** Refine response nuances based on clinician feedback, compile comprehensive API documentation, user guides, and system architecture reports. | Local / Hospital Server ($0) |
| **Phase 5** | February (กุมภาพันธ์) | **Final Defense & Live Demonstration:** Re-commission Cloud GPU instance for thesis defense demonstration, live system walkthrough via the testing interface, and committee examination. | Cloud GPU (NVIDIA RTX 4090 24GB VRAM) ($11.10 ~ 390 THB for 15 hrs) |

---

### 1.7 Budgetary Requirements
The system utilizes open-source framework components (model runtime engines, vector stores, and orchestration libraries) to eliminate proprietary software licensing costs. The operational budget is allocated for high-performance Cloud GPU infrastructure (NVIDIA RTX 4090) to support intensive model evaluation and live demonstration:

| Expense Item | Service Provider / Hardware Specification | Time Quota / Units | Cost (USD) | Cost (THB) |
| :--- | :--- | :---: | :---: | :---: |
| **Local Development** (Oct) | Local Workstation (Entry-level Laptop GPU / CPU) | Unlimited | $0.00 | 0 บาท |
| **Cloud GPU Testing** (Nov) | Cloud GPU Instance (NVIDIA RTX 4090 24GB VRAM, 5 vCPUs, 31GB RAM) | 50 Hours ($0.74/hr) | $37.00 | ~1,300 บาท |
| **Hospital Server** (Dec–Jan) | Naresuan University Hospital Server / Internal Network | Project Duration | $0.00 | 0 บาท |
| **Cloud GPU Defense** (Feb) | Cloud GPU Instance (NVIDIA RTX 4090 24GB VRAM - Rehearsal & Defense) | 15 Hours ($0.74/hr) | $11.10 | ~390 บาท |
| **Total Project Budget** | **Summary of all operational phases** | **65 Hours Cloud GPU** | **$48.10** | **~1,690 บาท** |

---

### 1.8 AI Disclosure Statements
We disclose that Google Gemini and AI-assisted development tools were utilized during the drafting of this report for grammatical refinement, formal English terminology selection, and stylistic consistency. The underlying RAG architecture, knowledge base formulation, evaluation methodologies, and experimental benchmarks were independently conceptualized, engineered, and executed by the author. All clinical statements have been verified against established oncology nursing literature and international CTCAE guidelines.

---
---

# Chapter 2: Theoretical Background and Principles

### 2.1 Chemotherapy Adverse Effects and CTCAE Grading Standard
Chemotherapy administration aims to eradicate malignant cells by interfering with cell division, DNA replication, or metabolic pathways. However, because cytotoxic compounds circulate systemically, healthy tissues with high physiological renewal rates—notably the oral mucosa, gastrointestinal tract, and peripheral nervous system—sustain collateral damage.

#### 2.1.1 Clinical Complications in Outpatient Oncology
1. **Oral Mucositis (แผลในช่องปากและเยื่อบุอักเสบ):** Cytotoxic agents induce cellular death in the basal epithelium, leading to painful mucosal inflammation, erythema, and deep ulceration. Patients experience severe oral burning sensations, intolerance to normal solid diet, dysphagia (swallowing impairment), and heightened risk of secondary systemic bacteremia.
2. **Chemotherapy-Induced Constipation (ท้องผูก):** Commonly administered chemotherapeutic regimens, coupled with routine supportive antiemetic medications (e.g., 5-HT3 receptor antagonists) and opioid analgesics, severely suppress enteric neuromuscular motility. Prolonged constipation can rapidly progress to severe fecal impaction, stercoral ulceration, and life-threatening bowel perforation.
3. **Chemotherapy-Induced Peripheral Neuropathy - CIPN (ปลายประสาทอักเสบ/ชา):** Neurotoxic agents (such as Taxanes and Platinum coordination complexes) cause mitochondrial injury and axonal degeneration in peripheral sensory nerves. Symptoms characteristically manifest bilaterally in the hands and feet ("stocking-glove distribution"), presenting as numbness, burning sensations, loss of fine touch, and impaired fine motor coordination (such as difficulty buttoning clothes or handwriting).

#### 2.1.2 Common Terminology Criteria for Adverse Events (CTCAE v5.0)
The National Cancer Institute (NCI) CTCAE v5.0 provides a globally standardized descriptive terminology and severity rating scale for cancer adverse events:
* **Grade 1 (Mild):** Asymptomatic or mild symptoms; clinical or diagnostic observations only; therapeutic intervention not indicated.
* **Grade 2 (Moderate):** Minimal, local, or non-invasive intervention indicated; limiting age-appropriate instrumental Activities of Daily Living (ADL) such as meal preparation, grocery shopping, or medication management.
* **Grade 3 (Severe):** Severe or medically significant but not immediately life-threatening; hospitalization or prolongation of existing hospitalization indicated; disabling; limiting self-care ADL (bathing, dressing, undressing, feeding, using the toilet).
* **Grade 4 (Life-threatening):** Urgent intervention indicated; consequences are immediately life-threatening.
* **Grade 5:** Death related to the adverse event.

---

### 2.2 Small Language Models (SLMs) and Localized Inference

#### 2.2.1 Evolution from Transformer Architecture
Modern generative language models are founded upon the autoregressive decoder-only Transformer architecture, employing multi-head self-attention mechanisms to dynamically capture semantic dependencies across distant tokens:
$$	ext{Attention}(Q, K, V) = 	ext{softmax}\left(rac{QK^T}{\sqrt{d_k}}ight)V$$
While large commercial foundation models demonstrate impressive conversational fluency, their application in hospital clinical workflows is constrained by significant vulnerabilities:
1. **Hallucination and Medical Safety:** Parametric memory models are susceptible to generating non-factual statements, incorrect dosages, or fabricated contraindications with high superficial confidence.
2. **Patient Data Privacy & Regulatory Compliance:** Transmitting unencrypted, sensitive medical queries to third-party commercial cloud APIs presents severe compliance risks under HIPAA and national personal data protection regulations (such as Thailand's PDPA).

#### 2.2.2 Small Language Models (SLMs) and Localized Serving
Small Language Models (ranging from 1.5B to 14B parameters) provide an optimal balance between cognitive reasoning and computational economy:
* **Multilingual Linguistic Proficiency:** Modern SLM architectures incorporate enlarged token vocabularies and multilingual pre-training corpora, achieving outstanding comprehension and token efficiency in non-Latin scripts, particularly formal and colloquial Thai.
* **Local Runtime Serving:** Local serving frameworks enable optimized 4-bit and 8-bit quantized model inference on commodity GPUs or dedicated hospital intranet servers, completely eliminating external network dependencies and ensuring total data privacy.

---

### 2.3 Retrieval-Augmented Generation (RAG) Architecture

#### 2.3.1 Principle of RAG
The RAG architecture overcomes the limitations of static parametric memory by decoupling knowledge storage from language generation. Instead of relying solely on parameters learned during pre-training, an external retrieval mechanism queries an indexed knowledge base to gather verified document snippets relevant to the user query. The language model then generates its response conditioned upon both the query and the retrieved context:

$$	ext{Output} \sim P(	ext{Response} \mid 	ext{Query}, 	ext{Context}_{	ext{retrieved}})$$

#### 2.3.2 Document Ingestion and Recursive Text Splitting
Clinical guidelines possess structured hierarchies (clinical categories, threshold grades, contraindications). Arbitrary character splitting risks bifurcating vital medical conditions. The system utilizes recursive character chunking with parameterized boundaries:
* **Chunk Size:** Configured to accommodate complete clinical grade definitions and self-care recommendations within a single chunk.
* **Chunk Overlap:** Preserves semantic continuity between consecutive chunks, preventing critical medical context from being severed at chunk boundaries.

#### 2.3.3 Dual-Stream Segregated Retrieval Strategy
Standard single-stream RAG retrieval frequently encounters "context imbalance": general self-care advice passages can dominate the top-$k$ nearest neighbors, displacing exact CTCAE grading threshold definitions. To resolve this challenge, the system employs a **Dual-Stream Segregated Retrieval Strategy**:
* **Stream 1 (Clinical Severity Criteria - $k_{	ext{grading}}$):** Filters chunks dedicated exclusively to standardized CTCAE severity grading rules.
* **Stream 2 (Supportive Care Advice - $k_{	ext{advice}}$):** Fetches chunks from evidence-based palliative nursing and supportive care manuals.
* Both retrieved context streams are systematically concatenated into a structured prompt, guaranteeing that the language model receives both diagnostic classification criteria and actionable home-care advice.

---

### 2.4 Vector Embeddings and Vector Databases

#### 2.4.1 Dense Multilingual Semantic Representations
Colloquial patient expressions (e.g., *"แสบปากกินข้าวไม่ได้"* or *"มือชาหยิบช้อนหล่น"*) exhibit significant lexical variation from formal clinical terms (*Oral Mucositis Grade 2* or *Peripheral Neuropathy Grade 2*). Lexical matching (such as BM25) fails to bridge this gap. Dense embedding models project textual units into high-dimensional latent vectors ($\mathbb{R}^d$):
* **Multilingual Dense Embeddings:** Trained specifically on cross-lingual, multi-granularity corpora, dense embeddings place colloquial symptom descriptions in close geometric proximity to their corresponding clinical definitions.

#### 2.4.2 Vector Database Management and Similarity Metric
An embedded vector database organizes high-dimensional vectors using Hierarchical Navigable Small World (HNSW) graph indices, enabling sub-millisecond approximate nearest neighbor (ANN) retrieval based on Cosine Similarity:
$$	ext{Cosine Similarity}(u, v) = rac{u \cdot v}{\|u\|_2 \|v\|_2}$$

---

### 2.5 Prompt Engineering and Medical Safety Guardrails

#### 2.5.1 Role-Based Grounding
System prompts establish an explicit clinical persona: *"Expert Oncology Medical AI Assistant"*. The prompt enforces rigid grounding directives:
* All factual statements must be derived exclusively from the provided context.
* If an inquiry falls outside the retrieved clinical context, the system is strictly instructed to respond: *"ขออภัย ข้อมูลในคู่มือไม่มีระบุเรื่องนี้ กรุณาปรึกษาแพทย์ประจำตัว"*

#### 2.5.2 Medical Escalation Protocol (Red-Flags)
To guarantee patient safety:
* **Urgent Clinical Referral:** Whenever user inputs describe Grade 3 or Grade 4 symptoms (e.g., inability to drink liquids for $>24$ hours, severe abdominal distention without bowel sounds or flatus, severe muscle weakness/falling), the model must output a prominent emergency notification urging immediate hospital care.
* **Prescription Restrictions:** The model is strictly prohibited from prescribing prescription-only medications or advising changes to ongoing chemotherapy treatment regimens.

---

### 2.6 Service-Oriented Backend Architecture and Hospital Integration Strategy

#### 2.6.1 Decoupled Headless Backend and API Integration with Hospital Chatbot
In enterprise healthcare environments, deploying an independent, standalone patient app introduces friction for patients already accustomed to established hospital digital channels. To resolve this, the architecture is engineered as a **Decoupled Headless Backend Service**:
* **Core Backend Intelligence:** The system encapsulates data ingestion, vector indexing, dual-stream retrieval, and LLM inference within an independent microservice.
* **Interoperability with Naresuan University Hospital Chatbot:** The backend exposes standardized RESTful API endpoints (such as `/api/v1/chemo-assess`). When a patient sends a symptom query through the hospital's existing conversational platform, the hospital gateway forwards the query payload to this RAG backend. The backend processes the inquiry, resolves the CTCAE grade, generates validated advice with safety guardrails, and returns a structured JSON payload to be rendered inside the hospital chatbot interface.
* **On-Premise Network Containment:** By deploying the backend server inside the hospital’s internal network, all patient communications and medical inferences remain completely within the firewall of Naresuan University Hospital, ensuring strict institutional compliance with patient data privacy standards.

#### 2.6.2 Streamlit as an Experimental Testbed and Administrative Prototype
During the engineering and evaluation lifecycle, a rapid web interface is required for clinical benchmarking, latency measurement, and researcher oversight. Streamlit is utilized strictly as an **Internal Experimental Testbed**:
* **Developer and Clinician Sandbox:** Provides an interactive testing ground where oncology specialists and AI researchers can simulate patient interactions, inspect raw retrieved context chunks, and verify guardrail behaviors without impacting live hospital systems.
* **Dynamic Parameter Optimization:** Facilitates live calibration of retrieval depth sliders ($k_{	ext{grading}}$ and $k_{	ext{advice}}$) and temperature settings, enabling the identification of the optimal hyperparameter configuration prior to production API deployment.
* **Performance Instrumentation:** Displays real-time response latency metrics (seconds) and memory utilization, serving as an operational benchmark for local versus Cloud GPU inference performance.

#### 2.6.3 Caching and State Management in Experimental Validation
* **Singleton Resource Caching:** Heavy runtime components (the dense embedding model and vector database) are instantiated once as singleton resources during testbed initialization, avoiding repetitive disk I/O and memory re-allocation.
* **Session State Management:** Maintains conversational context across multi-turn exchanges during clinician testing sessions, allowing comprehensive evaluation of context retention and safety consistency.

---

# References

1. National Cancer Institute. (2017). *Common Terminology Criteria for Adverse Events (CTCAE) Version 5.0*. U.S. Department of Health and Human Services, National Institutes of Health.
2. Lewis, P., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *Advances in Neural Information Processing Systems (NeurIPS 2020)*, 33, 9459-9474.
3. Touvron, H., et al. (2023). Llama 2: Open Foundation and Fine-Tuned Chat Models. *arXiv preprint arXiv:2307.09288*.
4. Qwen Team. (2024). *Qwen2.5: A Party of Foundation and Instruction-Tuned Models*. Technical Report.
5. Chen, J., et al. (2024). BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation. *arXiv preprint arXiv:2402.03216*.
6. LangChain Developers. (2024). *LangChain: Building Applications with LLMs through Composability*. https://python.langchain.com
7. Chroma Team. (2024). *Chroma: The AI-Native Open-Source Embedding Database*. https://docs.trychroma.com
8. RunPod Inc. (2024). *RunPod: Globally Distributed Cloud GPU Infrastructure for AI/ML Workloads*. https://www.runpod.io
9. American Cancer Society. (2023). *Chemotherapy Side Effects and Management: A Guide for Patients and Families*. Atlanta, GA.
10. Asian Pacific Journal of Cancer Prevention. (2021). Constipation in Chemotherapy Patients: A Diagnostic Accuracy Study. *APJCP / PMC8850898*, 22(8), 2541-2548.
