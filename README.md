# Fashion Recommendation Engine - AI/ML Assignment

This repository contains a complete multi-modal fashion recommendation system built for the Flutrr AI/ML Engineer assignment. The system combines machine learning, vector databases, and retrieval-augmented generation (RAG) to provide intelligent fashion recommendations.

## Project Overview

The solution implements a multi-modal recommendation engine using fine-tuned CLIP embeddings and LangChain-powered RAG system for natural language fashion advice. The system processes both product images and text descriptions to create semantic embeddings, stores them in a vector database, and generates personalized fashion recommendations through natural language queries.

## Notebooks Description

### 1. Data_prep_fashion.ipynb
This notebook handles the initial data preprocessing and preparation for the fashion dataset.

- Loads the Fashion Product Images Dataset from Kaggle with error handling for malformed entries
- Categorizes fashion items into logical groups (topwear, bottomwear, footwear, accessories)
- Creates structured product descriptions combining attributes like gender, category, and color
- Handles missing values and standardizes product information across different categories
- Generates positive and negative text pairs for contrastive learning
- Implements data validation checks to ensure consistency in product attributes
- Creates category mappings for topwear, bottomwear, and accessory groups
- Simulates price ranges based on product categories for realistic pricing
- Exports cleaned dataset with positive descriptions and negative samples
- Provides statistical analysis of product distribution across categories
- Ensures compatibility with downstream multi-modal training requirements

### 2. 2-finetune-clip-sbert-assign-final_final last.ipynb
This notebook implements the core machine learning component by fine-tuning a CLIP model for fashion-specific embeddings.

- Loads pre-trained CLIP ViT-L-14 model using sentence-transformers framework
- Creates triplet datasets with anchor images, positive descriptions, and negative examples
- Implements parameter freezing strategy to train only projection layers efficiently
- Configures contrastive learning using MultipleNegativesRankingLoss for alignment
- Sets up custom evaluators for measuring cosine similarity and Recall@k metrics
- Implements batch processing with optimized learning rates and regularization
- Trains model using SentenceTransformerTrainer with evaluation at each epoch
- Monitors training progress with validation splits across train/valid/test sets
- Achieves 75% Recall@1 performance on fashion product retrieval tasks
- Saves model checkpoints with versioning for reproducible deployment
- Deploys final model to HuggingFace Hub as dejasi5459/clip-fashion-embeddings-final-10k-ft
- Demonstrates significant improvements over base CLIP model for fashion similarity

### 3. embeddings_generation.ipynb
This notebook focuses on generating embeddings using the fine-tuned model and creating the vector database infrastructure.

- Loads fine-tuned CLIP model from HuggingFace with fallback to base model
- Processes fashion dataset to create comprehensive product descriptions
- Implements batch embedding generation for both text and image modalities
- Initializes ChromaDB as persistent vector database for product embeddings
- Creates collection with metadata support for filtering and search operations
- Normalizes embeddings and validates consistency across the entire dataset
- Preserves rich metadata including categories, prices, colors, and seasonal attributes
- Implements efficient search functions with metadata filtering capabilities
- Conducts quality checks through sample similarity queries and clustering analysis
- Benchmarks query response times and embedding quality metrics
- Creates production-ready database with backup and recovery procedures
- Provides search interface for text queries, image queries, and filtered searches

### 4. langchain-app.ipynb
This notebook implements the complete RAG system using LangChain for natural language fashion recommendations.

- Integrates fine-tuned CLIP model with HuggingFace embeddings for LangChain compatibility
- Creates custom FashionProductRetriever class for seamless ChromaDB integration
- Implements LangGraph workflow for structured query processing and response generation
- Uses GPT-4.1-mini as the language model for generating personalized fashion advice
- Supports natural language queries like "What should I wear for a brunch date?"
- Implements metadata filtering by gender, category, style preferences, and price ranges
- Creates fashion-specific prompt templates for contextual styling advice
- Builds retrieval pipeline with query embedding, vector search, and product retrieval
- Supports user preference specification for personalized demographic targeting
- Generates detailed explanations with styling tips and outfit coordination advice
- Provides interactive examples covering casual to formal occasion dressing scenarios
- Implements system evaluation metrics for recommendation quality and response times
- Delivers production-ready RAG pipeline for intelligent fashion advisory services

## Technical Stack

- **Machine Learning**: PyTorch, sentence-transformers, HuggingFace Transformers
- **Vector Database**: ChromaDB for embedding storage and similarity search
- **RAG Framework**: LangChain with LangGraph for workflow orchestration  
- **Language Model**: OpenAI GPT-4.1-mini for fashion advice generation
- **Data Processing**: Pandas, NumPy, PIL for dataset preparation
- **Model Deployment**: HuggingFace Hub for model versioning and distribution

## Dataset

The project uses the Fashion Product Images Dataset (Small) from Kaggle, containing product images, descriptions, categories, and metadata for comprehensive fashion understanding.

## Key Features

- Multi-modal embeddings combining text and image features
- Semantic similarity search with metadata filtering
- Natural language query processing for fashion recommendations
- Personalized advice generation based on user preferences
- Scalable vector database architecture for production deployment
- Integration with modern LLM frameworks for enhanced user experience
