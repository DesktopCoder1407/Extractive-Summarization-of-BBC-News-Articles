summary1_text = "neymar scored his side's second goal with a curling free kick, and 15 minutes to play in the 2-2 draw at sevilla on saturday night, according to reports in spain."
summary2_text = "barcelona's neymar substituted in 2-2 draw at sevilla on saturday night, spain's kamui kobayashi claims a late free kick in the champions league after his second goal with the score."

reference_text = "neymar was taken off with barcelona 2-1 up against sevilla. the brazil captain was visibly angry, and barca went on to draw 2-2. neymar has been replaced 15 times in 34 games this season. click here for all the latest barcelona news."

summary1_tokens = summary1_text.replace(',', '').replace('.', '').split()
summary2_tokens = summary2_text.replace(',', '').replace('.', '').split()
reference_tokens = reference_text.replace(',', '').replace('.', '').split()

def make_bigram(tokens):
    output_bigram = list()
    for i in range(len(tokens) - 1):
        output_bigram.append(tokens[i] + " " + tokens[i + 1])
    return output_bigram

def ROGUE_1():
    s1_overlap = 0
    for token in summary1_tokens:
        if token in reference_tokens:
            s1_overlap += 1

    s2_overlap = 0
    for token in summary2_tokens:
        if token in reference_tokens:
            s2_overlap += 1

    ## RECALL: overlapping / words in reference summary
    ## PRECISION: overlapping / words in system summary
    s1_recall = s1_overlap/len(reference_tokens)
    s1_precision = s1_overlap/len(summary1_tokens)
    s2_recall = s2_overlap/len(reference_tokens)
    s2_precision = s2_overlap/len(summary2_tokens)

    print(f"S1 Recall: {s1_overlap}/{len(reference_tokens)}")
    print(f"S1 Precision: {s1_overlap}/{len(summary1_tokens)}")
    print(f"S1 F1-Score: {2 * (s1_precision*s1_recall)/(s1_precision+s1_recall)}")
    print()
    print(f"S2 Recall: {s2_overlap}/{len(reference_tokens)}")
    print(f"S2 Precision: {s2_overlap}/{len(summary2_tokens)}")
    print(f"S2 F1-Score: {2 * (s2_precision*s2_recall)/(s2_precision+s2_recall)}")


def ROGUE_2():
    summary1_bigrams = make_bigram(summary1_tokens)
    summary2_bigrams = make_bigram(summary2_tokens)
    reference_bigrams = make_bigram(reference_tokens)

    s1_overlap = 0
    for token in summary1_bigrams:
        if token in reference_bigrams:
            s1_overlap += 1

    s2_overlap = 0
    for token in summary2_bigrams:
        if token in reference_bigrams:
            s2_overlap += 1

    ## RECALL: overlapping / words in reference summary
    ## PRECISION: overlapping / words in system summary
    s1_recall = s1_overlap/len(reference_bigrams)
    s1_precision = s1_overlap/len(summary1_bigrams)
    s2_recall = s2_overlap/len(reference_bigrams)
    s2_precision = s2_overlap/len(summary2_bigrams)

    print(f"S1 Recall: {s1_overlap}/{len(reference_bigrams)}")
    print(f"S1 Precision: {s1_overlap}/{len(summary1_bigrams)}")
    print(f"S1 F1-Score: {2 * (s1_precision*s1_recall)/(s1_precision+s1_recall) if s1_overlap != 0 else 0}")
    print()
    print(f"S2 Recall: {s2_overlap}/{len(reference_bigrams)}")
    print(f"S2 Precision: {s2_overlap}/{len(summary2_bigrams)}")
    print(f"S2 F1-Score: {2 * (s2_precision*s2_recall)/(s2_precision+s2_recall)if s2_overlap != 0 else 0}")


print(2 * ((3/7)*(3/8))/((3/7)+(3/8)))

