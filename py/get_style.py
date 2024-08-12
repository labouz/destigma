# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, TextClassificationPipeline
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import string
from lexicalrichness import LexicalRichness
# call get_emotion function
from get_emotion import get_emotion
# openai client
import json
import openai
# with open('../data/secrets.json') as f:
#     secrets = json.load(f)

# api_key = secrets['OPENAI_API_KEY_SR']

# client = openai.Client(api_key=api_key)

# Load the GoEmotion model and tokenizer
# tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions", model_max_length=512, padding=True, truncation=True)
# model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")
# classifier = pipeline(task="text-classification", 
#                       model=model, 
#                       tokenizer=tokenizer, 
#                       truncation=True,
#                       max_length=512,
#                       top_k=3)
# def classify_emotion(text):
#     # inputs = tokenizer(text, return_tensors="pt", max_length=512, padding=True, truncation=True)
#     results = classifier(text)
#     sorted_results = sorted(results[0], key=lambda x: x['score'], reverse=True)[:3]
#     return sorted_results

def mtld_LD(word_list):
    def mtld_calc(words, ttr_threshold=0.72):
        count, total = 0, 0
        types = set()
        for word in words:
            types.add(word)
            ttr = len(types) / (total + 1)
            if ttr <= ttr_threshold:
                count += 1
                types = set()
                total = 0
            else:
                total += 1
        if total > 0: count += total / len(words)  # Fractional part
        return count
    
    length_fwd = mtld_calc(word_list)
    length_bwd = mtld_calc(word_list[::-1])
    
    return (length_fwd + length_bwd) / 2

# Modified analyze_text_llm function
def analyze_text_llm(text, client):
    # Tokenize sentences and words
    sentences = sent_tokenize(text)
    # words = word_tokenize(text)

    # Emotion analysis using GoEmotion
    # top_3_emotions = classify_emotion(text)
    # [emotion['label'] for emotion in blah[0]]
    # emotion_descriptions = ", ".join([emotion['label'] for emotion in top_3_emotions])
    emotion = get_emotion(text, client)
    # Punctuation analysis
    punctuation_counts = {key: text.count(key) for key in string.punctuation}
    common_punctuation = ', '.join([p for p, count in punctuation_counts.items() if count > 0])

    # Active vs. Passive Voice
    def is_passive(sentence):
        tagged = pos_tag(word_tokenize(sentence))
        passive = False
        for i in range(len(tagged) - 1):
            if tagged[i][1] in ['was', 'were'] and tagged[i+1][1] == 'VBN':
                passive = True
        return passive

    passive_sentences = sum(is_passive(sentence) for sentence in sentences)
    passive_voice_usage = "none" if passive_sentences == 0 else "some"

    # Sentence length variability
    sentence_lengths = [len(s.split()) for s in sentences]
    min_length = min(sentence_lengths)
    max_length = max(sentence_lengths)
    average_length = sum(sentence_lengths) / len(sentence_lengths)

    # Lexical diversity
    # lexical_diversity_score = len(set(words)) / len(words)
    # lexical_diversity_description = "moderately high" if lexical_diversity_score > 0.5 else "low"
    # refine to use MTLD
    # lexical_diversity = mtld_LD(words)
    lex = LexicalRichness(text)
    lex = lex.mtld(threshold=0.72)


    results = {
        "top emotions": emotion,
        "punctuation_usage": f"moderate, with {common_punctuation} being most frequent",
        "passive_voice_usage": passive_voice_usage,
        "sentence_length_variation": f"ranging from short ({min_length} words) to long ({max_length} words) with an average of {average_length:.1f} words per sentence",
        # "lexical_diversity": f"{lexical_diversity:.2f} (MTLD) or {lex:.2f} (MTLD)"
        "lexical_diversity": f"{lex:.2f} (MTLD)"
    }

    return results