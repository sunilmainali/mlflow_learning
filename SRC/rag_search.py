from vector_stores import query_vector_store
from models import get_llm_model
import mlflow
from dotenv import load_dotenv
import os
load_dotenv()
mlflow_url = os.getenv("mlflow_url")
# MLflow setup
mlflow.set_tracking_uri(mlflow_url)
mlflow.set_experiment("Langchain_mlflow")
mlflow.openai.autolog()
mlflow.langchain.autolog()


def calculate_cost(usage, model_name: str):
    """
    Calculate cost based on model pricing.
    """
    PRICING = {
        "gpt-4o-mini": {
            "input": 0.15 / 1_000_000,
            "output": 0.60 / 1_000_000,
        }
    }

    price = PRICING.get(model_name)
    if not price:
        return 0.0

    return (
        usage["prompt_tokens"] * price["input"]
        + usage["completion_tokens"] * price["output"]
    )


def generate_answer(query: str):
    # Retrieve documents
    relevant_docs = query_vector_store(query, top_k=3)
    context = "\n".join(doc.page_content for doc in relevant_docs)

    prompt = f"""
Context:
{context}

Question: {query}
Answer:
"""

    llm = get_llm_model()
    response = llm.invoke(prompt)

    # Token usage (LangChain standard)
    usage = response.response_metadata.get("token_usage", {})
    model_name = response.response_metadata.get("model_name", "gpt-4o-mini")

    cost = calculate_cost(usage, model_name)

    return response.content, relevant_docs, cost


@mlflow.trace
def chat_completion(query: str, user_id: str, session_id: str):
    mlflow.update_current_trace(
        metadata={
            "mlflow.trace.user": user_id,
            "mlflow.trace.session": session_id,
            "app.name": "rag_chat",
        }
    )

    mlflow.log_param("query", query)

    answer, retrieved_docs, cost = generate_answer(query)

    mlflow.log_metric("retrieved_docs", len(retrieved_docs))
    mlflow.log_metric("query_cost_usd", cost)
    mlflow.log_param("final_total_cost_usd", cost)

    return answer


if __name__ == "__main__":
    answer = chat_completion(
        query="What is the attention mechanism in transformers?",
        user_id="user_123",
        session_id="session_456",
    )

    print("Answer:", answer)
