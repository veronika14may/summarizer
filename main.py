import json
from summarizer import summarize_batch, MAX_CHARS
from rouge_metrics import evaluate

def load_data_from_json(filepath: str):
    """Считывает тексты и эталоны из JSON-файла"""
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    texts = [item["text"] for item in data]
    references = [item["reference"] for item in data]
    return texts, references

def print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)

def print_results(
    texts: list[str],
    summaries: list[str],
    references: list[str] | None = None,
    max_chars: int = MAX_CHARS,
) -> None:
    print_separator("═")
    print(f"  АВТОМАТИЧЕСКАЯ СУММАРИЗАЦИЯ  (макс. {max_chars} символов)")
    print_separator("═")

    for i, (text, summary) in enumerate(zip(texts, summaries), 1):
        print(f"\n{'[ Документ ' + str(i) + ' ]':^70}")
        print_separator()
        print(f"\nРеферат ({len(summary)} симв.):")
        print(f"  {summary}")
        if references:
            ref = references[i - 1]
            print(f"\nЭталон ({len(ref)} симв.):")
            print(f"  {ref}")
        print_separator()

    if references:
        print("\n" + "═" * 70)
        print("  МЕТРИКИ ROUGE  (лемматизация Pymorphy2)")
        print("═" * 70)

        report = evaluate(summaries, references)
        for rouge_name, scores in report.items():
            print(f"\n{rouge_name}:")
            print(f"  Macro Precision : {scores['precision']:.4f}")
            print(f"  Macro Recall    : {scores['recall']:.4f}")
            print(f"  Macro F1        : {scores['f1']:.4f}")
            print()
            print(f"  {'Документ':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
            print(f"  {'-'*42}")
            for j, doc_scores in enumerate(scores["per_doc"], 1):
                print(
                    f"  Документ {j:<3}  "
                    f"{doc_scores['precision']:>10.4f} "
                    f"{doc_scores['recall']:>10.4f} "
                    f"{doc_scores['f1']:>10.4f}"
                )
        print_separator("═")


def main() -> None:
    sample_texts, sample_references = load_data_from_json("dataset.json")
    summaries = summarize_batch(sample_texts, max_chars=MAX_CHARS)
    print_results(sample_texts, summaries, sample_references)

if __name__ == "__main__":
    main()