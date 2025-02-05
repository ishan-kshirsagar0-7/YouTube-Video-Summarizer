# YouTube Video Summarizer

## Introduction

This project presents a sophisticated Python application for summarizing YouTube Videos. It combines the power of a quantized GGUF LLaMa-2 model (llama-2-13b-chat.Q4_K_M.gguf), a RAG-based system using ChromaDB and Langchain, and the youtube-transcript-api. The application is designed to process, analyze, and condense content from YouTube videos into concise summaries.

## Methodology

**GGUF LLaMa-2 Model**: Employed a quantized version of the LLaMa-2 model, optimized for efficient processing and accurate summarization.

**RAG-based System with ChromaDB**: Integrated a Retrieval-Augmented Generation (RAG) system using ChromaDB to enhance the summarization process.

**Langchain Integration**: Leveraged Langchain for its powerful NLP capabilities, aiding in effective language understanding and generation.

**YouTube Transcript API**: Utilized youtube-transcript-api to fetch the textual content of YouTube videos for processing.

## Demo

TRY IT OUT : [CLICK HERE](https://huggingface.co/spaces/unpairedelectron07/YT_Video_Summarizer)

**Note** : The demo on HuggingFace takes a while to generate summary, so if you need faster results, clone this repository, download the "llama-2-13b-chat.Q4_K_M.gguf" model from HuggingFace and run "app.py" locally.

## Outputs

![ytvsop1](https://github.com/user-attachments/assets/b1c0d884-997a-42f9-8eb5-50c5724aace0)
The video in the above screenshot is [Every Programming Language Ever Explained in 15 Minutes](https://www.youtube.com/watch?v=ajIcjx0PeYU) by **Flash Bytes**.

![ytvsop2](https://github.com/user-attachments/assets/00f2dac5-946c-42b0-a580-5bf2f1648da0)
The video in the above screenshot is [What The Prisoner's Dilemma Reveals About Life, The Universe, and Everything](https://www.youtube.com/watch?v=mScpHTIi-kM&t=3s) by **Veritasium**.

We can clearly see that this tool managed to summarize even a 30 minute long video in under 5 bullet points.

## Key Steps

**1. Library Installation**: Installation of essential libraries including the quantized GGUF LLaMa-2 model, Langchain, ChromaDB, and Gradio.\
**2. GGUF LLaMa-2 Model Setup**: Initialization and configuration of the GGUF LLaMa-2 model for transcript summarization.\
**3. Transcript Extraction**:  Utilization of youtube-transcript-api to retrieve YouTube video transcripts.\
**4. RAG Based Summarization**: Implementation of a RAG system with ChromaDB for efficient and accurate content summarization.\
**5. User Interface**: Creation of a Gradio-based interactive interface for easy user interaction and summarization requests.\
**6. PDF Generation**: Converting the summarized content into PDF format for user accessibility and convenience.

## Results

The application demonstrates its capability to fetch, process, and summarize YouTube video transcripts effectively. The integration of advanced AI models and systems results in high-quality summaries that capture the essence of the video content.

## Conclusion

This project exemplifies the successful application of cutting-edge AI technologies in creating a practical and user-friendly tool for video content summarization. The combination of the GGUF LLaMa-2 model, RAG-based systems, and other AI components provides a robust solution for processing and summarizing vast amounts of video data efficiently.
