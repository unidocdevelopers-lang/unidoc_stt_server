from fastapi import FastAPI
from pydantic import BaseModel
from rapidfuzz import process, fuzz
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import csv
import json
import nltk
from nltk.corpus import words, stopwords
import spacy


nlp = spacy.load("en_core_web_sm")

load_dotenv()

try:
    nltk.data.find("corpora/words")
except LookupError:
    nltk.download("words")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")


english_words = set(w.lower() for w in words.words())
common_stopwords = set(w.lower() for w in stopwords.words("english"))

app = FastAPI()

def load_corrections_from_csv(filepath="corrections.csv"):
    corrections = {}
    try:
        with open(filepath, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                incorrect = row.get("incorrect", "").strip().lower()
                correct = row.get("correct", "").strip()
                if incorrect and correct:
                    corrections[incorrect] = correct
    except FileNotFoundError:
        print(f" corrections.csv not found at: {filepath}")
    return corrections


def is_potential_medical_term(word: str):
    return word.lower() in corrections.keys()


def save_incorrect_word(word: str):
    word = word.strip().lower()
    filepath = "wrong_words.json"
    try:
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []

        if word not in data:
            data.append(word)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f" Error saving incorrect word: {word} → {e}")

corrections = load_corrections_from_csv()

def get_named_entities(text: str):
    doc = nlp(text)
    return set(ent.text.lower() for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"])

def smart_split_by_prefix(word, corrections, threshold=85):
    """
    Split the word into two only if BOTH parts (prefix and suffix) are in the corrections dictionary.
    Avoid splitting if either part is not a valid correction key.
    """
    word = word.lower()
    for i in range(3, len(word) - 2):
        prefix = word[:i]
        suffix = word[i:]

        # Exact match
        if prefix in corrections and suffix in corrections:
            return f"{corrections[prefix]} {corrections[suffix]}"

        # Fuzzy match
        prefix_match = process.extractOne(prefix, corrections.keys(), scorer=fuzz.ratio)
        suffix_match = process.extractOne(suffix, corrections.keys(), scorer=fuzz.ratio)

        if (
            prefix_match and prefix_match[1] >= threshold and prefix_match[0] in corrections and
            suffix_match and suffix_match[1] >= threshold and suffix_match[0] in corrections
        ):
            return f"{corrections[prefix_match[0]]} {corrections[suffix_match[0]]}"

    return None



def correct_medical_with_dose(word, corrections, threshold=85):
    match = re.match(r"^([a-zA-Z\-]+)(z)?(\d+)(mg|ml|mcg|g|units)?([a-zA-Z]*)?$", word)
    if not match:
        return None

    med_part, _, dose_num, dose_unit, suffix = match.groups()
    lower_med = med_part.lower()

    if lower_med in corrections:
        corrected_med = corrections[lower_med]
    else:
        # ✅ Only split if smart_split finds both parts in corrections
        smart_split = smart_split_by_prefix(lower_med, corrections, threshold=threshold)
        if smart_split:
            corrected_med = smart_split
        else:
            corrected_med = lower_med

    dose_str = f"{dose_num}{dose_unit or ''}"
    return f"{corrected_med} {dose_str} {suffix or ''}".strip()


def correct_transcript(text, corrections, threshold=85):
    named_entities = get_named_entities(text)

    def correct_word(word):
        try:
            match = re.match(r"(\W*)([\w\-]+)(\W*)", word)
            if not match:
                return word

            prefix, core, suffix = match.groups()
            lower_core = core.lower()

            # Step 1: Dose match (e.g., "paracitamol500mg")
            dose_fixed = correct_medical_with_dose(core, corrections, threshold)
            if dose_fixed:
                return f"{prefix}{dose_fixed}{suffix}"

            # Step 2: Skip numbers (doses, years)
            if any(char.isdigit() for char in core):
                return f"{prefix}{core}{suffix}"

            # Step 3: Exact match from corrections
            if lower_core in corrections:
                corrected = corrections[lower_core]
   
                if corrected.lower() == lower_core:
                    return f"{prefix}{core}{suffix}"
                return f"{prefix}{corrected}{suffix}"


            # Step 4: Apply fuzzy only if likely to be medical
            result = process.extractOne(lower_core, corrections.keys(), scorer=fuzz.ratio)
            if result and result[1] >= threshold:
                corrected = corrections[result[0]]
                return f"{prefix}{corrected}{suffix}"

            # Step 5: Save unknown words
            if (
                lower_core not in english_words
                and lower_core not in common_stopwords
                and lower_core not in named_entities
            ):
                smart_split = smart_split_by_prefix(lower_core, corrections)
                if smart_split:
                    print(f"[✅ Auto-split]: {lower_core} → {smart_split}")
                    return f"{prefix}{smart_split}{suffix}"
                save_incorrect_word(lower_core)

            return f"{prefix}{core}{suffix}"

        except Exception as e:
            print(f" Word correction error: {word} → {e}")
            return word  # fallback

    return ' '.join(correct_word(w) for w in text.split())


class CorrectionInput(BaseModel):
    text: str

@app.post("/correct")
async def correct_text(data: CorrectionInput):
    corrected = correct_transcript(data.text, corrections)
    return {"original": data.text, "corrected": corrected}
