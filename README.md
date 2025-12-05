# 🧠 MemContext: The Next-Gen Multi-modal Agent Memory Architecture

<div align="center">

![License](https://img.shields.io/badge/license-Apache%202.0-blue)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)
![Multi-modal](https://img.shields.io/badge/Modality-Audio%20|%20Video%20|%20Image%20|%20Document%20|%20Text-green)

**The First "Omni-Modal" Memory Framework for AI Agents**

*Store Everything · Search Everything · Frame-Level Precision*

[中文文档](README_CN.md) • [Core Features](#-core-features) • [Architecture](#-architecture) • [Use Cases](#-use-cases)

</div>

---

## 📖 Introduction

**MemContext is designed to build a persistent, high-fidelity, and evolutionary "Second Brain" for AI Agents.**

Most existing Memory frameworks forcibly "flatten" the rich physical world into pure text, resulting in the loss of visual details and spatio-temporal context dislocation. **MemContext refuses this dimensional reduction.**

We are a **Multi-modal Native** memory framework engineered to ingest video, audio, and document streams in their raw, high-fidelity forms. Whether it's hundreds of hours of raw footage or a subtle visual cue in the background of a video, MemContext delivers a closed loop from **Omni-modal Ingestion** and **Stream Storage** to **Frame-level Retrieval**.

We are not building a static database; we are building **Native Spatio-Temporal Perception** for Agents.

---

## ✨ Core Features

### 1. ♾️ Omni-Modal Storage & Retrieval
**Break the Text-to-Text barrier.** MemContext unifies the processing paradigm for heterogeneous data.
* **Unified Input:** Seamlessly handles Video, Audio, Image, Documents, and Text.
* **Native Multi-modal Indexing:** Utilizes a "Parallel Retrieval Workflow" to index visual and auditory signals directly via vector embeddings, rather than relying solely on generated text descriptions/captions.
* **Cross-Modal Search:** Supports "Search Video by Image" and "Search Text by Audio."

### 2. 🎞️ Infinite Stream Processing
**Forget about Context Window Anxiety.**
* Designed specifically for **Long-Context** and **24/7 Continuous** Agents.
* Supports ingestion of infinite-length video and continuous audio streams.
* **Dynamic Chunking:** Built-in **Continuity Check** mechanisms ensure memory context never breaks, even when processing massive meeting records, TV series, or surveillance streams.

### 3. 🎯 0.1s Spatio-Temporal Precision
**Eliminate Timestamp Hallucinations.**
* Traditional RAG/Memory systems only tell you "which file" the answer is in. MemContext tells you **"which minute and which second."**
* **SOTA Precision:** Retrieves and locates video/audio segments with **0.1-second accuracy**.
* **Value:** Returns precise, 0.1s key snippets (The Needle) rather than bloated 1GB files (The Haystack).

---

## 🏗️ Architecture

MemContext draws inspiration from human cognitive processes and OS storage structures, adopting a tiered storage architecture paired with a dual-path retrieval engine.

<div align="center">
  <img src="docs/MemContext_Workflow_v0.png" alt="MemContext Architecture Workflow" width="100%">
</div>

### 🧠 Memory Lifecycle
Data flows through a biomimetic processing pipeline:
1.  **Ingestion:** A unified processor extracts multi-dimensional features from heterogeneous sources.
2.  **STM (Short-Term Memory):** Handles immediate context streams, performing embedding calculations and preliminary filtering.
3.  **MTM (Medium-Term Memory):** A session-based buffer introducing a **Heat Calculation Algorithm**. The system dynamically evaluates data value based on Visit Frequency ($N_{visit}$), Interaction Depth ($L_{interaction}$), and Recency ($R_{recency}$).
4.  **LTM (Long-Term Memory):** High-heat information "crystallizes" into permanent storage (User Persona & Knowledge Base), while low-value data is pruned from working memory.

### 🔍 Precision Search Engine
Supports natural language, image, and video segment queries:
1.  **Semantic Filtering:** Rapidly screens relevant session contexts from MTM.
2.  **Vector Similarity Matching:** Performs deep scans within the LTM Knowledge Base.
3.  **Spatio-Temporal Timestamping:** Executes high-precision frame-level positioning to lock onto specific data fragments.
4.  **Result Aggregation:** Fuses retrieved video clips, background knowledge, and dialogue history into a structured **Context Dictionary** for the Agent.

---

## 🚀 Use Cases

### ✂️ Intelligent Video Editing Co-Pilot
* **Scenario:** A documentary editor faces TBs of B-Roll and needs to find a shot of "the protagonist laughing by the sea with seagull sounds in the background."
* **The MemContext Solution:**
    * **Rapid Screening:** Uses semantic understanding to lock onto files containing "sea" and "laugh."
    * **Surgical Precision:** Combines Audio Modality (seagull recognition) and Visual Modality (protagonist laughing) with **0.1s precision** to output exact `Inpoint` and `Outpoint` timecodes.
    * **Result:** The Agent generates a ready-to-use Edit Decision List (EDL), eliminating manual seeking.

### 🖼️ Next-Gen Context-Aware Gallery
* **Scenario:** A user wants to recall a memory, not just a file: "Find photos from 2 years ago when we discussed the startup plan at that cafe playing jazz."
* **The MemContext Solution:**
    * **Cross-Modal Association:** A complex compound query. The system retrieves image content (Cafe), ambient audio (Jazz stream), and dialogue records (Startup plan text).
    * **Memory Evocation:** Links fragmented images with the sound and conversation of that moment, reconstructing the scene rather than just returning a static JPEG.

### 🕵️‍♂️ Legal AI & Evidence Analysis
* **Scenario:** An Agent needs to find contradictions in 50 hours of witness testimony video.
* **The MemContext Solution:** The Agent queries specific actions or statements (e.g., "Find every time the suspect looked at their watch"). The system returns video slices precise to 0.1s for immediate verification.

### 🎓 Online Education & Knowledge Extraction
* **Scenario:** A student wants to review the specific derivation of the "Attention Mechanism in Transformers" from a 100-hour semester course.
* **The MemContext Solution:**
    * **Infinite Stream:** Indexes the entire semester's video.
    * **Knowledge Localization:** Instead of watching the whole lecture, the system jumps to the **exact 2-minute fragment** where the professor draws the formula on the blackboard, synced with the speech transcript.

### 👓 Wearable AI & Episodic Memory
* **Scenario:** A smart glass user asks, "Where did I leave my AirPods?"
* **The MemContext Solution:** Indexes Ego-centric video streams in real-time. Using **Visual Object Grounding**, it searches for the visual features of "AirPods" (without relying on text tags), backtracking to the last interaction frame to provide a location snapshot.

---

## ⚡ Quick Start

*(Codebase is being polished and will be released shortly)*

## 🤝 Contributing

MemContext is under active development. If you are interested in Multi-modal RAG, Agent Memory Systems, or Vector Search, please Star this repo and follow our progress.

## 📜 License

Apache 2.0