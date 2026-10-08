# Enterprise Ticket Transformer

### From Transformer Fundamentals to Enterprise-Ready LLM Applications

An educational AI engineering project for IT support ticket classification, built with natural language processing (NLP), Transformer architectures, and a planned Large Language Model (LLM) extension.

The working application compares seven ticket classifiers (V1–V7) and serves a selected model through a FastAPI service (V8). Prioritization, team routing, sentiment analysis, summarization, and AI-assisted responses are proposed extensions.

> **Project Status:** Lecture 1 classification models V1–V7 and the V8 API are implemented. Lecture 2 LLM packages contain a roadmap and scaffolding; they do not generate responses yet. All included ticket data is synthetic and does not establish enterprise performance.

---

## 1. Project Overview

Enterprise IT organizations manage large volumes of support tickets across cloud infrastructure, DevOps, application support, networking, and service management.

Traditional ticket management workflows often depend on manual classification, prioritization, and assignment.

**Enterprise Ticket Transformer** currently provides a modular ticket classification learning project. The wider ticket intelligence platform described here is the project vision.

The project also serves as a structured learning environment for understanding the evolution of NLP into modern Transformer and LLM architectures.

### Primary Objectives

- Understand and implement core NLP and Transformer concepts.
- Compare and serve ticket classification models.
- Explore modern LLM architectures and inference techniques.
- Implement automated ticket prioritization and routing.
- Introduce AI-powered ticket summarization and response suggestions.
- Evaluate model quality on consistent splits and representative data.
- Apply enterprise software engineering and deployment practices.

---

## 2. Problem Statement

Enterprise IT support teams receive thousands of service requests and incident tickets.

These tickets often contain unstructured descriptions, technical terminology, incomplete information, and varying levels of urgency.

### Key Challenges

| Challenge | Business Impact |
|---|---|
| Manual ticket classification | Increased operational effort |
| Incorrect ticket assignment | Delayed incident resolution |
| Inconsistent prioritization | Critical incidents may be overlooked |
| Repetitive ticket analysis | Reduced support engineer productivity |
| Long ticket conversations | Increased investigation time |
| Limited intelligent automation | Higher operational costs |

Traditional rule-based approaches may struggle to interpret the context and intent of complex technical support requests.

A context-aware AI solution can help support teams process tickets more consistently and efficiently.

---

## 3. Proposed Solution

Extend the existing classifier and API into a ticket intelligence platform. The current API accepts ticket text and returns a classification label and scores; the other capabilities in this section are planned.

### Core Capabilities

#### 3.1 Automated Ticket Classification

The implemented models classify the included samples into four labels:

- Access
- Hardware
- Network
- Software

Future datasets could support more specific categories such as:

- Cloud Infrastructure
- Network Operations
- Application Support
- DevOps and CI/CD
- Database Services
- Security Operations
- General IT Support

#### 3.2 Intelligent Priority Detection

Planned: analyze ticket descriptions and recommend priority levels. No priority model or endpoint is implemented.

Example:

```json
{
  "ticket_id": "INC-10001",
  "description": "Production application is unavailable",
  "category": "Application Support",
  "priority": "Critical"
}
```

This JSON illustrates a future result, not the current API response. Priority recommendations would need validated predictions and configurable business rules.

#### 3.3 Intelligent Ticket Routing

Planned: recommend a support team based on:

- Ticket category
- Technical context
- Predicted urgency
- Business routing rules

#### 3.4 Sentiment Analysis

Identify customer sentiment and potential frustration to assist support teams in handling sensitive incidents.

This capability is not implemented.

#### 3.5 AI-Powered Ticket Summarization

Use an LLM to summarize lengthy support conversations into concise incident descriptions.

This capability is not implemented.

#### 3.6 AI-Assisted Response Generation

Generate suggested responses that support engineers can review, modify, and approve before sending.

This capability is not implemented. The [Lecture 2 roadmap](docs/lecture_02_llm/README.md) requires human approval before any external write.

---

## 4. High-Level Architecture

```text
                   Current workflow
              Ticket text (POST /predict)
                           |
                           v
              FastAPI validation + API key
                           |
                           v
              Selected V1–V7 classifier
                           |
                           v
             Label + scores + score type

                   Planned extensions
        Priority / sentiment / team routing
                           |
                           v
           LLM summary + response draft
                           |
                           v
                    Human approval
                           |
                           v
                    ITSM integration
```

The current API serves one selected local model artifact. The planned stages have no endpoint or ITSM integration yet. See [API usage](#8-getting-started) and the [Lecture 2 roadmap](docs/lecture_02_llm/README.md).

---

## 5. Learning and Implementation Roadmap

The project follows a progressive learning approach. Lecture 1 is implemented; Lecture 2 is planned.

### Lecture 1 — Transformers

**Objective:** Understand the evolution of NLP and implement the fundamental components of Transformer architectures.

| Module | Topic | Practical Implementation |
|---|---|---|
| 1.1 | NLP Fundamentals | Text preprocessing |
| 1.2 | Tokenization | Tokenization experiments |
| 1.3 | Embeddings | Token embedding visualization |
| 1.4 | Word2Vec | Semantic representation baseline |
| 1.5 | RNN/LSTM | Sequence classification baseline |
| 1.6 | Attention Mechanisms | Attention implementation |
| 1.7 | Transformer Architecture | Transformer-based ticket classifier |

**Implemented Deliverable:**

V1–V7 ticket classifiers, including a scratch Transformer encoder and pretrained BERT variants, plus the V8 FastAPI service. See [why each generation exists](docs/generations.md). Team routing is not implemented.

### Lecture 2 — Large Language Models

**Objective:** Explore modern LLM architectures and implement efficient inference techniques.

| Module | Topic | Practical Implementation |
|---|---|---|
| 2.1 | LLM Architectures | Encoder/decoder architecture comparison |
| 2.2 | Mixture of Experts | Lightweight MoE implementation |
| 2.3 | MHA/MQA/GQA | Attention mechanism comparison |
| 2.4 | Rotary Position Embeddings | RoPE implementation |
| 2.5 | Context Length | Context window and KV-cache experiments |
| 2.6 | Sampling Strategies | Temperature, top-k and top-p decoding |
| 2.7 | Integration | Unsent ticket-response draft with human approval |

**Expected Deliverable:**

An evaluated LLM-powered ticket summarization and AI-assisted response drafting prototype. This deliverable is not implemented yet.

### Future Learning Modules

Potential future extensions include:

- LLM supervised fine-tuning (classifier fine-tuning exists in V7)
- LoRA and parameter-efficient fine-tuning
- LLM evaluation and benchmarking (classifier evaluation exists)
- Inference optimization
- Retrieval-Augmented Generation
- AI agent integration
- Production monitoring

These modules will be introduced incrementally.

---

## 6. Technology Stack

### AI and Machine Learning

- Python 3.12
- PyTorch
- Hugging Face Transformers
- Word2Vec
- RNN/LSTM
- Transformer architectures

The Word2Vec, LSTM, and scratch Transformer implementations are in `src/`. LLM implementation work is planned; Hugging Face Datasets and scikit-learn are not project dependencies.

### Backend Development

- FastAPI
- Pydantic
- REST APIs

### Testing and Engineering

- Python `unittest`
- Git and GitHub
- GitHub Actions
- Docker
- Configuration management

### Enterprise AI Concepts

- Self-attention
- Multi-head attention
- Multi-query attention
- Grouped-query attention
- Rotary Position Embeddings
- Mixture of Experts
- KV caching
- Sampling strategies
- Model evaluation

> Current dependencies are pinned in [`requirements.txt`](requirements.txt) and [`requirements-dev.txt`](requirements-dev.txt). In the concepts listed above, multi-query attention, grouped-query attention, RoPE, MoE, KV caching, and sampling are planned; self-attention and classification evaluation are implemented.

---

## 7. Repository Structure

```text
Enterprise_Ticket_Transfomer/
│
├── .github/                # GitHub workflows and configuration
├── api/                    # REST API implementation
├── configs/                # Application and model configurations
├── data/                   # Datasets and preprocessing outputs
├── docs/                   # Technical documentation
├── models/                 # Model artifacts
├── reports/                # Evaluation and experiment reports
├── scripts/                # Utility and execution scripts
├── src/                    # Application source code
├── tests/                  # Automated tests
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── requirements-dev.txt
└── requirements.txt
```

Local `.venv/` and `.hf_cache/` directories may appear after setup; both are ignored by Git.

### Proposed Source Code Organization

```text
src/
├── preprocessing/
├── embeddings/
├── baselines/
├── transformer/
├── llm/
│   ├── architectures/
│   ├── moe/
│   ├── attention/
│   ├── positional_encoding/
│   ├── context/
│   ├── sampling/
│   └── inference/
├── training/
└── evaluation/
```

The Lecture 1 directories contain working code. The `src/llm/` subdirectories exist as empty package scaffolding for Lecture 2; [`src/llm/README.md`](src/llm/README.md) tracks their intended use. Shared inference code is in `src/inference/`.

---

## 8. Getting Started

### Prerequisites

- Python 3.12
- Git
- A Python virtual environment and dependencies from `requirements-dev.txt` for tests

### Clone the Repository

```bash
git clone https://github.com/karitselmuthu/Enterprise_Ticket_Transfomer.git

cd Enterprise_Ticket_Transfomer
```

### Create a Virtual Environment

```bash
python3.12 -m venv .venv

source .venv/bin/activate
```

### Install Dependencies

```bash
python -m pip install -r requirements.txt
```

For development dependencies:

```bash
python -m pip install -r requirements-dev.txt
```

### Run Tests

```bash
python -m unittest discover -s tests -v
```

CI uses the same test command. To train the included synthetic examples and run local inference:

```bash
python -m src.training.train                         # V1
python -m src.training.train_v2                      # V2
python -m src.training.train_neural --version v3     # LSTM
python -m src.training.train_neural --version v4     # LSTM + attention
python -m src.training.train_neural --version v5     # Scratch Transformer
python -m src.inference.predict --model models/v5.json "VPN disconnects"
```

V6/V7 require a local pretrained checkpoint. `prepare_pretrained` downloads the default checkpoint once; use its `--model-id` and `--revision` options to select and pin an approved alternative.

```bash
python -m src.training.prepare_pretrained
python -m src.training.train_pretrained --version v6
python -m src.training.train_pretrained --version v7
python -m src.evaluation.compare_all
```

To serve a trained V1 artifact:

```bash
TICKET_API_KEY=change-me TICKET_MODEL_PATH=models/v1.json \
  uvicorn api.main:app --host 127.0.0.1 --port 8000 --no-access-log
```

In a second terminal:

```bash
curl -H 'X-API-Key: change-me' -H 'Content-Type: application/json' \
  -d '{"text":"VPN disconnects during meetings"}' http://127.0.0.1:8000/predict
```

The API requires an API key by default. Its response has `version`, `label`, `scores`, and `score_type`; it does not return priority, sentiment, or a draft.

---

## 9. Example Use Case

### Input Ticket

```text
VPN disconnects during meetings.
```

### Illustrative Expected Output

```json
{
  "version": "v1",
  "label": "network",
  "scores": {
    "access": 0.0847,
    "hardware": 0.0957,
    "network": 0.7320,
    "software": 0.0875
  },
  "score_type": "model_probability"
}
```

This shows the current `/predict` response shape, using a locally trained V1 sample artifact; scores are rounded. Outputs depend on the selected model and dataset. Model scores should not be treated as calibrated confidence without validation. The API does not generate the proposed fields in Section 3.

---

## 10. Model Evaluation

The existing evaluation code compares classification models on held-out IDs from a locked dataset split. The sample tickets are synthetic, so these results verify the workflow rather than enterprise accuracy. See the [sample comparison](reports/sample_error_analysis.md) and [application comparison](reports/application_comparison.md).

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1-score
- Per-class support and mistake IDs

### LLM Evaluation

- Response relevance
- Summary quality
- Factual consistency
- Human evaluation
- Inference latency
- Memory utilization

These are planned evaluation criteria for Lecture 2; no LLM evaluation runs are implemented yet.

### Baseline Comparison

The implemented classification comparison covers:

```text
V1: Token counts + Naive Bayes
      |
      v
V2: Word2Vec + class centroids
      |
      v
V3–V4: LSTM, then LSTM + attention
      |
      v
V5: Scratch Transformer encoder
      |
      v
V6–V7: Frozen, then fine-tuned pretrained BERT
```

Run `python -m src.evaluation.compare_all` after training all seven stages. Their small synthetic holdouts do not support a production model choice. A separate, de-identified real-data evaluation and promotion process is described in [real-ticket evaluation and model promotion](docs/real_data_and_promotion.md). LLM-based processing remains future work.

---

## 11. Enterprise Engineering Principles

The project uses some enterprise engineering practices as teaching examples. Additional safeguards are required before using it with real tickets.

### Modularity

Separate preprocessing, model training, inference, evaluation, and API components.

### Configuration Management

Maintain reusable configuration files for experiments and deployment environments.

### Testing

Run `unittest` checks and a data audit in CI. Add focused tests as each Lecture 2 phase is implemented.

### Observability

Structured logging, inference metrics, and production monitoring are planned; they are not provided by the current API.

### Security

The API validates input, requires an API key by default, and requires a promotion manifest in production mode. Real ticket data must be de-identified and reviewed before use; the current service is not a complete production security design.

### Human Oversight

The planned response workflow requires human approval before customer communications or external ticket writes. No response drafting or approval workflow is implemented yet.

### Reproducibility

The current split manifests record dataset hashes and held-out IDs. Model artifacts record their version; use consistent splits and review evaluation results before comparing models.

---

## 12. Development Roadmap

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Repository foundation | Implemented |
| Phase 2 | NLP and Transformer fundamentals | Implemented in Lecture 1 |
| Phase 3 | Transformer ticket classification | Implemented through V1–V7 |
| Phase 4 | LLM architecture experiments | Planned; package scaffolding exists |
| Phase 5 | Ticket summarization | Planned |
| Phase 6 | AI-assisted response generation | Planned |
| Phase 7 | API and ITSM integration | Classification API implemented; ITSM integration planned |
| Phase 8 | Evaluation and optimization | Synthetic classification comparisons implemented; real-data and LLM evaluation planned |
| Phase 9 | Deployment and monitoring | Dockerfile present; production deployment and monitoring planned |

The repository currently has no live deployment or verified enterprise performance. See the [current root-level workflow](#8-getting-started) for runnable commands.

---

## 13. Expected Business Benefits

The solution aims to demonstrate the potential for:

- Reduced manual ticket processing.
- More consistent ticket classification.
- Improved support team assignment.
- Faster identification of critical incidents.
- Reduced repetitive analysis.
- Better support engineer productivity.
- Improved consistency of customer communications.

Actual business improvements must be validated through testing and operational metrics.

---

## 14. Project Vision

The long-term goal is to develop a modular, reliable, and extensible enterprise AI platform that combines Transformer-based understanding with LLM-powered assistance.

The project is also intended to provide practical experience in:

**NLP → Transformers → LLM Architectures → Fine-Tuning → Evaluation → Enterprise AI Engineering**

---

## 15. License

Refer to the [LICENSE](LICENSE) file for licensing information.

---

**Enterprise Ticket Transformer**

*Building practical enterprise AI systems while learning the foundations of modern Transformer and LLM architectures.*
