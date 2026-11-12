
import os
import sys
import csv
import argparse

# Add project root to Python import path.
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)
sys.path.insert(0, PROJECT_ROOT)

from src.retrieval import PolicyRetriever


def page_ranges_overlap(start1, end1, start2, end2):
    """Check whether two page ranges overlap."""
    return start1 <= end2 and start2 <= end1


def is_correct_result(result, question):
    """Check document ID and page-range correctness."""
    metadata = result.get("metadata", {})

    if metadata.get("doc_id") != question["gold_doc_id"]:
        return False

    try:
        retrieved_start = int(metadata.get("page_start", 0))
        retrieved_end = int(metadata.get("page_end", 0))

        gold_start = int(question["gold_page_start"])
        gold_end = int(question["gold_page_end"])
    except (TypeError, ValueError):
        return False

    return page_ranges_overlap(
        retrieved_start,
        retrieved_end,
        gold_start,
        gold_end
    )


def load_questions(path):
    """Load and validate evaluation questions."""
    with open(path, "r", encoding="utf-8-sig", newline="") as file:
        questions = list(csv.DictReader(file))

    if not questions:
        raise ValueError("No evaluation questions found.")

    required_columns = {
        "question_id",
        "question",
        "gold_doc_id",
        "gold_page_start",
        "gold_page_end"
    }

    missing = required_columns - set(questions[0].keys())

    if missing:
        raise ValueError(
            f"Missing CSV columns: {', '.join(sorted(missing))}"
        )

    for question in questions:
        if not question["question"].strip():
            raise ValueError(
                f"Empty question: {question['question_id']}"
            )

        for field in (
            "gold_page_start",
            "gold_page_end"
        ):
            try:
                int(question[field])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid {field} in {question['question_id']}"
                )

    return questions


def evaluate():
    parser = argparse.ArgumentParser(
        description="Evaluate PolicyLens retrieval performance."
    )

    parser.add_argument(
        "--collection",
        default="policylens_final_30docs",
        help="ChromaDB collection name."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of documents retrieved per question."
    )

    args = parser.parse_args()

    if args.top_k < 10:
        raise ValueError(
            "Use top_k >= 10 to calculate Recall@10 correctly."
        )

    eval_directory = os.path.dirname(os.path.abspath(__file__))

    questions_path = os.path.join(
        eval_directory,
        "questions.csv"
    )

    if args.collection == "policylens_final_30docs":
        output_suffix = "30docs"
    elif args.collection == "policylens_fixed_30docs":
        output_suffix = "fixed_30docs"
    else:
        output_suffix = args.collection

    metrics_path = os.path.join(
        eval_directory,
        f"metrics_{output_suffix}.csv"
    )

    retrieval_log_path = os.path.join(
        eval_directory,
        f"retrieval_log_{output_suffix}.csv"
    )

    # Load questions.
    questions = load_questions(questions_path)

    total_questions = len(questions)

    print("\n========================================")
    print("       POLICYLENS RETRIEVAL EVALUATION")
    print("========================================")
    print(f"Collection: {args.collection}")
    print(f"Total questions: {total_questions}")
    print(f"Top K: {args.top_k}")
    print("----------------------------------------")

    # Initialize retriever.
    retriever = PolicyRetriever(
        collection_name=args.collection
    )

    recall_3_hits = 0
    recall_5_hits = 0
    recall_10_hits = 0
    reciprocal_rank_total = 0.0

    retrieval_logs = []

    # Evaluate every question.
    for question in questions:

        query = question["question"]

        results = retriever.search_policies(
            query=query,
            top_k=args.top_k
        )

        first_correct_rank = None

        for rank, result in enumerate(results, start=1):

            correct = is_correct_result(
                result,
                question
            )

            if correct and first_correct_rank is None:
                first_correct_rank = rank

            metadata = result.get("metadata", {})

            retrieval_logs.append({
                "question_id": question["question_id"],
                "rank": rank,
                "question": query,
                "retrieved_doc_id": metadata.get("doc_id", ""),
                "retrieved_title": metadata.get("title", ""),
                "retrieved_page_start": metadata.get("page_start", ""),
                "retrieved_page_end": metadata.get("page_end", ""),
                "score": result.get("score", ""),
                "is_correct": correct
            })

        # Calculate Recall@K and MRR.
        if first_correct_rank is not None:

            if first_correct_rank <= 3:
                recall_3_hits += 1

            if first_correct_rank <= 5:
                recall_5_hits += 1

            if first_correct_rank <= 10:
                recall_10_hits += 1

            reciprocal_rank_total += (
                1.0 / first_correct_rank
            )

        status = (
            f"Rank {first_correct_rank}"
            if first_correct_rank is not None
            else "Not found"
        )

        print(
            f"{question['question_id']}: {status}"
        )

    # Calculate final metrics.
    recall_3 = recall_3_hits / total_questions
    recall_5 = recall_5_hits / total_questions
    recall_10 = recall_10_hits / total_questions

    mrr = reciprocal_rank_total / total_questions

    # Save aggregate metrics.
    metrics = [
        {"metric": "Total Questions", "value": total_questions},
        {"metric": "Recall@3", "value": round(recall_3, 4)},
        {"metric": "Recall@5", "value": round(recall_5, 4)},
        {"metric": "Recall@10", "value": round(recall_10, 4)},
        {"metric": "MRR", "value": round(mrr, 4)}
    ]

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=["metric", "value"]
        )

        writer.writeheader()
        writer.writerows(metrics)

    # Save detailed retrieval logs.
    log_fields = [
        "question_id",
        "rank",
        "question",
        "retrieved_doc_id",
        "retrieved_title",
        "retrieved_page_start",
        "retrieved_page_end",
        "score",
        "is_correct"
    ]

    with open(
        retrieval_log_path,
        "w",
        encoding="utf-8",
        newline=""
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=log_fields
        )

        writer.writeheader()
        writer.writerows(retrieval_logs)

    # Display results.
    print("\n========================================")
    print("             FINAL RESULTS")
    print("========================================")

    print(f"Total questions: {total_questions}")
    print(f"Recall@3:        {recall_3:.4f}")
    print(f"Recall@5:        {recall_5:.4f}")
    print(f"Recall@10:       {recall_10:.4f}")
    print(f"MRR:             {mrr:.4f}")

    print("\nEvaluation files saved:")
    print(metrics_path)
    print(retrieval_log_path)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    evaluate()
