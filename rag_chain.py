from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from prompt_template import format_prompt
from llm_wrapper import generate_answer

def run_rag(question, model):
    # Load vector store
    db = FAISS.load_local(
        "faiss_index",
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True
    )
    retriever = db.as_retriever(search_kwargs={"k": 3})
    
    # Get relevant docs and format prompt
    docs = retriever.get_relevant_documents(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = format_prompt(context, question)

    # Generate and return answer
    return generate_answer(model, prompt)

# Debug/test
if __name__ == "__main__":
    from llm_wrapper import load_llm
    model = load_llm()
    print(run_rag("What are the working hours?", model))
