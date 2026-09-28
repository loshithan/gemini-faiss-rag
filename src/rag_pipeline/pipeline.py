
from .embedder import embed_text
from .store import create_faiss_index, search_query_in_index
def answer_question(question, context):
    """
    Answer a question based on the provided context.

    Args:
        question (str): The question to be answered.
        context (str): The context in which to find the answer.

    Returns:
        str: The answer to the question.
    """
    # Combine the question and context into a single prompt
    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

    # Use the embedding model to generate an answer
    answer_embedding = embed_text(prompt)

    # Create a FAISS index for the context embeddings
    context_embedding = embed_text(context)
    index = create_faiss_index([context_embedding])

    # Search for the most similar embedding to the answer
    results = search_query_in_index(answer_embedding, index)

    # Return the most similar context as the answer
    if results:
        return context
    else:
        return "No relevant context found."

if __name__ == "__main__":
    # Example usage
    question = "What is the color of the sky?"
    context = "The sky is blue during the day and can appear red or orange during sunrise and sunset."

    answer = answer_question(question, context)
    print(f"Question: {question}\nAnswer: {answer}")