# LLM Hyperparameters Experiment

**A Streamlit app to run factorial experiments on LLM sampling hyperparameters and analyze readability via Flesch Reading Ease.**

## Features
- Configure Temperature, Top-p, Top-k, and replicates from the sidebar  
- Automated runs against OpenAI API (gpt-4o-mini)  
- Collects outputs, computes Flesch scores, fits three-way ANOVA  
- Diagnostic plots and interaction/box plots  

## Quickstart

```bash
git clone https://github.com/your-username/llm-hyperparam-expt.git
cd llm-hyperparam-expt
pip install -r requirements.txt
export OPENAI_API_KEY=<your_api_key>
streamlit run streamlit_app.py
```

## Docker

```bash
docker build -t llm-expt .
docker run -e OPENAI_API_KEY=<key> -p 8501:8501 llm-expt
```
