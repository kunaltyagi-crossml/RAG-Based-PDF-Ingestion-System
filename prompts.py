from langchain.prompts import PromptTemplate

# RAG Prompt Template for Question Answering
rag_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an intelligent assistant specialized in answering questions based on provided documents.

Your task is to analyze the given context and provide accurate, helpful answers to the user's question.

Guidelines:
- Answer ONLY based on the information provided in the context below
- If the answer is not present in the context, clearly state: "I don't have enough information in the provided document to answer this question."
- Provide detailed and comprehensive answers when the information is available
- Use bullet points or structured formatting when it makes the answer clearer
- Quote relevant parts of the document when appropriate
- If the question is ambiguous, provide the best possible interpretation

Context:
{context}

Question:
{question}

Answer:
"""
)

# Alternative prompt for summarization tasks
summarization_prompt = PromptTemplate(
    input_variables=["context"],
    template="""Provide a concise and comprehensive summary of the following text.

Focus on the main ideas, key points, and important details.

Text:
{context}

Summary:
"""
)

# Conversational prompt for follow-up questions
conversational_prompt = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template="""You are a helpful AI assistant engaging in a conversation about a document.

Previous conversation:
{chat_history}

Current context from document:
{context}

User question:
{question}

Provide a natural, conversational response that takes into account both the previous conversation and the current context.

Answer:
"""
)