# Smart Home Support Agent

A RAG-based AI agent that helps users troubleshoot smart home devices using natural language queries.

---

## Solo Project

**Igwilo Emmanuel**  ITAI2376

---

## Problem Statement

Smart home ecosystems (Alexa, Google Home, SmartThings, etc.) are widely used but difficult to troubleshoot. Users face connectivity failures, unresponsive devices, and confusing error states  and support documentation is often too technical or scattered across manufacturer websites.

**Target user:** Renters and homeowners who use smart home devices and want quick self-service support without reading lengthy manuals.

---

## Option Chosen

**Option A: Single AI Agent**

The agent was designed and planned in the Midterm and is now fully implemented.

---

## Architecture Overview

The agent follows a **ReAct (Reason + Act)** loop with two tools:

1. **category_tool** is the reasoning step. Before searching, the agent classifies the user's query into a device category (lighting, thermostat, camera, lock, etc.) using keyword matching. This is the "Reason" step.

2. **retrieval_tool** is the action step. Encodes the cleaned query using a SentenceTransformer model (`all-MiniLM-L6-v2`) into a 384-dimensional embedding, then searches a FAISS index for the top 3 most semantically similar known issues. This is the "Act" step.

The full loop is: **Observe ---> Reason (category_tool) ----> Act (retrieval_tool) ---> Respond**

A confidence threshold of 0.6 (cosine similarity) filters out weak matches so the agent returns "no confident match found" rather than a misleading answer.

The UI is built with Gradio and displays the suggested fix, related issues, and a full agent reasoning trace per query.


## Frameworks and Tools

| Component | Tool/Library |
|---|---|
| Embedding model | `sentence-transformers`  `all-MiniLM-L6-v2` |
| Vector search | `faiss-cpu`  `IndexFlatIP` (cosine similarity) |
| UI | `gradio` |
| Deep learning backend | `torch` (underlying SentenceTransformer) |
| Knowledge base | Custom JSON  40 smart home issue-resolution pairs |
| Reasoning pattern | ReAct (Reason + Act) |

---

## Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

No API keys are required. The agent runs entirely on local open-source models.

---

## How to Run

Open `agent.ipynb` in Jupyter or Google Colab and run all cells in order.

```bash
# Local Jupyter
jupyter notebook agent.ipynb
```

Or in Google Colab: upload `agent.ipynb`, then click **Runtime → Run all**.

The Gradio interface will launch automatically after the final cell. A local URL (e.g. `http://127.0.0.1:7860`) will appear in the output.

---

## Example Usage

**Example 1**
- Input: `"My Alexa won't respond to voice commands"`
- Category detected: `speaker`
- Output: Speak clearly and close to the microphone. Check your device language settings. Ensure your internet connection is stable.
- Confidence: 78%

**Example 2**
- Input: `"The camera keeps going offline"`
- Category detected: `camera`
- Output: Check Wi-Fi connection. Restart the camera. Ensure power supply is stable.
- Confidence: 91%

**Example 3**
- Input: `"Motion sensor keeps triggering false alerts"`
- Category detected: `sensor`
- Output: Solution: Check for vibrations or loose mounting of the sensor. Detail: Ensure the sensor and magnet are firmly attached to the door/window frame.
- Confidence: 83%

---

## Known Limitations

- **Small knowledge base:** The corpus contains 40 issue-resolution pairs. Queries outside these categories may fall below the confidence threshold and return no answer.
- **No memory:** Each query is independent. The agent does not remember previous turns in a conversation.
- **Static knowledge:** The knowledge base is a fixed JSON file. It does not update automatically when new device issues emerge.
- **No generative responses:** The agent retrieves stored resolutions rather than generating natural-language answers. Adding a generative LLM layer is a planned improvement.

---

## Demo
[Watch the demo here](https://www.youtube.com/watch?v=Sv9HnbvPnKc)

The demo shows the agent handling at least 3  scenarios end-to-end.

---

## Repository Structure

```
agent.ipynb              # Main notebook — run this
requirements.txt         # Python dependencies
.env.example             # Environment variable template (no keys needed)
.gitignore               # Git exclusions
README.md                # This file
data/
  smart_home_corpus.json # Knowledge base (40 issue-resolution pairs)
docs/
  architecture.png       # Architecture diagram
demo/
  demo.mp4               # Screen recording (or link above)
```

---

## Deep Learning Connections

**Transformers & Attention:** all-MiniLM-L6-v2 is a distilled Transformer encoder. It uses multi-head self-attention so every token in a query attends to every other token — capturing semantic equivalences like "won't turn on" ≈ "not responding" that bag-of-words models miss entirely.

**Representation Learning & Embeddings:** The model was trained with contrastive learning to cluster semantically similar sentences in a 384-dimensional embedding space. Retrieval quality depends entirely on the quality of these learned representations, making this a direct application of the representation learning module.

---
## Reflection

### What worked
The core RAG pipeline performed better than expected. Search using
**all-MiniLM-L6-v2** handled informal phrasing well, queries like "my bulb
keeps flickering" correctly matched "Light Not Turning On" without any exact
keyword overlap. The confidence threshold (0.6) was effective at preventing
the agent from returning weak or irrelevant matches, which was one of my
original concerns. The Gradio interface came together quickly
and the reasoning trace output made the agent's decision-making visible, which
helped during debugging.

### What did not work
The **extract_issues_from_guide()** function initially collapsed entire JSON
sections into single issue-resolution pairs, which meant multi-step resolutions
lost their structure. Query cleaning (**clean_text()**) was also applied to
resolution text before display, producing lowercase unpunctuated output that
looked unprofessional. Also, **category_tool** uses simple keyword matching, which
misclassifies queries with unusual phrasing  for example, "hub not syncing"
is not caught by any category keyword and defaults to general.

### Biggest technical challenge
The hardest problem I faced was that **clean_text()** was being applied to queries at
index-build time but not at query time, creating a mismatch between what was
stored in the FAISS index and what was being searched. Retrieval scores were
lower than they should have been because the query embeddings and index
embeddings were generated from differently formatted text. Fixing this required
me to trace the full pipeline from ingestion through retrieval to confirm where
normalization was and was not being applied.

### Path change from Midterm
The overall architecture stayed the same as the blueprint. The main change
was me adding a second tool (**category_tool**) to satisfy the two-tool requirement
and to implement an explicit ReAct reasoning step. The
embedding model was also upgraded from **paraphrase-MiniLM-L3-v2** to
**all-MiniLM-L6-v2** as planned in Week 2.

### What I would build next

**The most valuable next step would be replacing the stored resolution retrieval
with a generative LLM response. Expanding the knowledge base beyond 40 entries and adding
IndexIVFFlat for faster approximate search at scale (1000+ entries) would also
be proper as well.**
---
## Citations

- Hugging Face  all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- FAISS (Meta AI Research): https://github.com/facebookresearch/faiss
- Sentence Transformers docs: https://www.sbert.net/docs/sentence_transformer/pretrained_models.html
- Gradio docs: https://www.gradio.app/docs
