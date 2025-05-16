import json 


def main():
    file_name = "<extracted_overall_reasons_for_all_models.json>"
    overall_categories = [
        "Lacking empathy or emotional attunement",
        "Displaying an inappropriate tone or attitude (e.g., dismissive, superficial)",
        "Providing inaccurate suggestions (e.g., containing wrong information or making recommendations without sufficient evidence)",
        "Offering unconstructive feedback (e.g., lacking clarity or actionability)",
        "Demonstrating little personalization or relevance",
        "Containing language or terminology issues (e.g., typos, grammatical errors)",
        "Overgeneralizing or making judgments and assumptions without sufficient context"
    ]
    
    outputs= json.load(open(file_name))
    count_freq = {
        category: 0 for category in overall_categories
    }

    total_count = len(outputs['overall_category'])
    for idx, overall_out in enumerate(outputs['overall_category']):
        for category in overall_categories:
            if category in overall_out:
                count_freq[category] += 1

    count_freq['total'] = total_count

    total_percentage = {
        category: (count / total_count) for category, count in count_freq.items()
    }

    with open(file_name.replace(".json", "_percentage.json"), "w") as f:
        json.dump(total_percentage, f, indent=4)

    with open(file_name.replace(".json", "_count.json"), "w") as f:
        json.dump(count_freq, f, indent=4)

if __name__ == "__main__":
    main()