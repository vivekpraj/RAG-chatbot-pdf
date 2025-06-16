def format_prompt(context: str, question: str) -> str:
    return f"""
You are an AI assistant. You must answer the user's question **only** based on the given context. 
Do not repeat yourself. Do not generate multiple answers. Give a short and clear reply.

Context:
{context}

Question: {question}

Answer:
""".strip()
