import re
from abc import ABC, abstractmethod
from typing import Optional, Set

# Standard English stop words and conversational discourse markers to filter out
COMMON_STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd",
    "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there",
    "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll",
    "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's",
    "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't",
    "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    # Additional spoken narration filler words
    "today", "welcome", "hello", "let", "see", "look", "going", "just", "really", "also", "next",
    "like", "get", "got", "know", "think", "make", "take", "come", "want", "give", "well",
    "explore", "exploring", "discover", "discovering", "discuss", "discussing", "show", "showing",
    "learn", "learning", "understand", "understanding", "examine", "examining", "vast", "provide",
    "provides", "providing"
}


class BaseQueryExtractor(ABC):
    """Abstract interface for extracting visual B-roll search queries from scene narration."""

    @abstractmethod
    def extract_query(self, narration_text: str, visual_hint: Optional[str] = None) -> str:
        """Derives a concise 2-4 word stock footage search query."""
        pass


class RuleBasedQueryExtractor(BaseQueryExtractor):
    """
    Deterministic baseline query extractor.
    Uses frequency and part-of-speech heuristic filtering without requiring external LLM APIs.
    Priority is given to explicit user visual hints, followed by salience-filtered content tokens.
    """

    def extract_query(self, narration_text: str, visual_hint: Optional[str] = None) -> str:
        # If user explicitly supplied a visual hint, that takes precedence
        if visual_hint and visual_hint.strip():
            hint_clean = re.sub(r"[^\w\s-]", "", visual_hint.strip().lower())
            tokens = [t for t in hint_clean.split() if t]
            return " ".join(tokens[:4])

        # Clean narration text: strip punctuation and lowercase
        cleaned_text = re.sub(r"[^\w\s]", " ", narration_text.lower())
        raw_tokens = cleaned_text.split()

        # Filter out common stop words, numbers, and single characters
        meaningful_tokens = [
            token for token in raw_tokens
            if token not in COMMON_STOP_WORDS and len(token) > 2 and not token.isdigit()
        ]

        if not meaningful_tokens:
            # Fallback for short phrases composed primarily of stopwords (e.g. "Here we are")
            fallback_tokens = [t for t in raw_tokens if len(t) > 1]
            return " ".join(fallback_tokens[:3]) if fallback_tokens else "cinematic backdrop"

        # Select the top unique content tokens (up to 4) in order of appearance
        seen = set()
        selected_tokens = []
        for token in meaningful_tokens:
            if token not in seen:
                seen.add(token)
                selected_tokens.append(token)
            if len(selected_tokens) == 4:
                break

        return " ".join(selected_tokens)
