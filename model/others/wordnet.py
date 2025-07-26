from logging import Logger

import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus.reader.wordnet import Synset
from nltk.stem import WordNetLemmatizer


class WordnetModel:
    def __init__(
        self,
        logger: Logger | None = None,
    ):
        """
        Initialize wordnet model.
        """
        if logger:
            logger.info("Loading Wordnet model")
        self.is_concrete_noun_cache: dict[str, bool] = {}
        self.lemmatizer = WordNetLemmatizer()

    def _synsets(self, word: str) -> list[Synset]:
        """
        Get synsets for the given word.

        Args:
            word: The word to get synsets for.
        Returns:
            list: List of synsets containing the given word.
        """
        return wn.synsets(word, lang="eng")

    def get_synset_list(self, word: str) -> list[str]:
        """
        Use the WordNet library to get the synsets of a word and return a list of synonyms.

        Args:
            word: The word to get synsets for.
        Returns:
            list: List of synonyms for the given word.
        """
        return list({synset.lemmas()[0].name() for synset in self._synsets(word)})

    def is_concrete_noun(self, word: str) -> bool:
        """
        Check if the given word is a concrete noun.

        Args:
            word: The word to check, assumed to be a noun.
        """
        if " " in word:
            return any([self.is_concrete_noun(w) for w in word.split()])
        else:
            if word in self.is_concrete_noun_cache:
                return self.is_concrete_noun_cache[word]

            synsets: list[Synset] = wn.synsets(word, pos=wn.NOUN, lang="eng")
            if not synsets:
                self.is_concrete_noun_cache[word] = False
                return False

            _mayby_concrete = [
                "noun.person",
                "noun.artifact",
                "noun.object",
                "noun.animal",
                "noun.event",
                "noun.Tops",
                "noun.food",
                "noun.plant",
                "noun.substance",
            ]

            maynotby_concrete = [
                "noun.feeling",
                "noun.attribute",
                "noun.state",
                "noun.shape",
                "noun.time",
                "noun.quantity",
                "noun.cognition",
                "noun.event",
                "noun.communication",
                "noun.relation",
                "noun.act",
                "noun.location",
            ]
            # Determine whether it belongs to the category of concrete objects
            is_concrete = synsets[0].lexname() not in maynotby_concrete

            self.is_concrete_noun_cache[word] = is_concrete
            # return str(is_concrete) + '\t' + str(synsets[0].lexname())  # for test
            return is_concrete

    def lemma(self, word: str) -> str:
        """
        Lemmatization
        """
        return self.lemmatizer.lemmatize(word)

    def word_tokenize(self, text: str) -> list[str]:
        """
        Tokenization
        """
        return nltk.word_tokenize(text)
