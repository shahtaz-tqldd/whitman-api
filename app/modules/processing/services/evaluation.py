from app.modules.processing.models.embeddings import embed_texts


def main() -> None:
    vectors = embed_texts([
        "navy blue",
        "dark blue",
        "cotton",
        "100 percent cotton",
    ])

    print("--------NEW PRINT-----------")
    print(len(vectors))
    print(len(vectors[0]))
    print("-------------------")


if __name__ == "__main__":
    main()
