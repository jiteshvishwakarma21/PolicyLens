
import csv
import sys
from pathlib import Path

from src.hybrid_retrieval import HybridPolicyRetriever


QUESTIONS_FILE = Path("eval/questions.csv")
METRICS_FILE = Path("eval/metrics_hybrid.csv")
LOG_FILE = Path("eval/retrieval_log_hybrid.csv")

TOP_K = 10


def evaluate():
    if not QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"Questions file not found: {QUESTIONS_FILE}"
        )

    retriever = HybridPolicyRetriever(
        collection_name="policylens_structure_300_v1",
        model_name="all-MiniLM-L6-v2"
    )

    total_questions = 0
    recall_3 = 0
    recall_5 = 0
    recall_10 = 0
    reciprocal_rank_sum = 0

    log_rows = []

    with open(
        QUESTIONS_FILE,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        reader = csv.DictReader(file)

        for question in reader:
            question_id = question["question_id"]
            query = question["question"]
            gold_doc_id = question["gold_doc_id"]

            gold_start = int(question["gold_page_start"])
            gold_end = int(question["gold_page_end"])

            results = retriever.search(
                query,
                top_k=TOP_K,
                candidate_k=50
            )

            total_questions += 1
            first_correct_rank = None

            for result in results:
                metadata = result["metadata"]

                retrieved_doc_id = metadata.get("doc_id")
                page_start = metadata.get("page_start")
                page_end = metadata.get("page_end")

                try:
                    page_start = int(page_start)
                    page_end = int(page_end)
                except (TypeError, ValueError):
                    page_start = -1
                    page_end = -1

                is_correct = (
                    retrieved_doc_id == gold_doc_id
                    and page_start <= gold_end
                    and page_end >= gold_start
                )

                if is_correct and first_correct_rank is None:
                    first_correct_rank = result["rank"]

                log_rows.append({
                    "question_id": question_id,
                    "question": query,
                    "question_type": question.get("question_type", ""),
                    "rank": result["rank"],
                    "rrf_score": result["score"],
                    "retrieved_doc_id": retrieved_doc_id,
                    "title": metadata.get("title", ""),
                    "page_start": page_start,
                    "page_end": page_end,
                    "section_title": metadata.get("section_title", ""),
                    "is_relevant": is_correct,
                    "retrieved_text": result["document"][:200]
                })

            if first_correct_rank is not None:
                if first_correct_rank <= 3:
                    recall_3 += 1
                if first_correct_rank <= 5:
                    recall_5 += 1
                if first_correct_rank <= 10:
                    recall_10 += 1

                reciprocal_rank_sum += 1 / first_correct_rank

            print(
                f"{question_id}: "
                f"first relevant rank = {first_correct_rank}"
            )

    if total_questions == 0:
        raise ValueError("No questions found in questions.csv")

    metrics = {
        "retrieval_method": "Hybrid BM25 + Semantic + RRF",
        "total_questions": total_questions,
        "Recall@3": recall_3 / total_questions,
        "Recall@5": recall_5 / total_questions,
        "Recall@10": recall_10 / total_questions,
        "MRR": reciprocal_rank_sum / total_questions
    }

    with open(
        METRICS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    log_fields = [
        "question_id",
        "question",
        "question_type",
        "rank",
        "rrf_score",
        "retrieved_doc_id",
        "title",
        "page_start",
        "page_end",
        "section_title",
        "is_relevant",
        "retrieved_text"
    ]

    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=log_fields)
        writer.writeheader()
        writer.writerows(log_rows)

    print("\n========== HYBRID EVALUATION ==========")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

    print("\nSaved metrics:", METRICS_FILE)
    print("Saved retrieval log:", LOG_FILE)


if __name__ == "__main__":
    evaluate()
