from difflib import SequenceMatcher

def compare_texts(original, corrected):
    """
    Compares extracted handwriting with the corrected text and gives feedback.
    """
    similarity = SequenceMatcher(None, original, corrected).ratio()
    percentage = round(similarity * 100, 2)

    if percentage > 85:
        feedback = "Great job! Your handwriting closely matches the corrected text."
    elif 50 < percentage <= 85:
        feedback = "Good effort! Some words need improvement."
    else:
        feedback = "Needs improvement. Try to make your handwriting clearer."

    return {"similarity": percentage, "feedback": feedback}
