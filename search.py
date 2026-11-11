
import argparse
import sys

from src.retrieval import PolicyRetriever


def parse_filters(filter_values):
    """Convert key=value arguments into ChromaDB metadata filters."""
    filters = {}

    for item in filter_values:
        if "=" not in item:
            raise ValueError(
                f"Invalid filter '{item}'. Use key=value format."
            )

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key or not value:
            raise ValueError(f"Invalid filter: {item}")

        filters[key] = value

    return filters if filters else None


def main():
    parser = argparse.ArgumentParser(
        description="Search RBI policy documents using PolicyLens."
    )

    parser.add_argument(
        "query",
        type=str,
        help="Question or search query."
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of results to display (default: 5)."
    )

    parser.add_argument(
        "--filter",
        action="append",
        default=[],
        help="Metadata filter in key=value format. Can be repeated."
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top_k must be at least 1.")

    try:
        filters = parse_filters(args.filter)

        retriever = PolicyRetriever(
            collection_name="policylens_final_30docs"
        )

        results = retriever.search_policies(
            query=args.query,
            top_k=args.top_k,
            where=filters
        )

        print("\n" + "=" * 65)
        print("                  POLICYLENS SEARCH")
        print("=" * 65)
        print(f"Query: {args.query}")
        print(f"Results requested: {args.top_k}")
        print(f"Filters: {filters if filters else 'None'}")
        print("-" * 65)

        if not results:
            print("No matching documents found.")
            return

        for result in results:
            metadata = result["metadata"]

            print(f"\nRank: {result['rank']}")
            print(f"Similarity Score: {result['score']}")
            print(f"Title: {metadata.get('title', 'Unknown')}")
            print(f"Issuer: {metadata.get('issuer', 'Unknown')}")
            print(f"Document ID: {metadata.get('doc_id', 'Unknown')}")

            print(
                f"Pages: {metadata.get('page_start', 'Unknown')} - "
                f"{metadata.get('page_end', 'Unknown')}"
            )

            print(
                f"Section: "
                f"{metadata.get('section_title', 'Unknown')}"
            )

            print("\nExtracted Text:")
            print(result["document"][:500])
            print("-" * 65)

        print(f"\nTotal results: {len(results)}")

    except (ValueError, Exception) as error:
        print(f"\nSearch error: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
