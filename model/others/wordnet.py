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
        初始化 wordnet 模型。
        """
        if logger:
            logger.info("Loading Wordnet model")
        self.is_concrete_noun_cache: dict[str, bool] = {}
        self.lemmatizer = WordNetLemmatizer()

    def _synsets(self, word: str) -> list[Synset]:
        """
        获取给定单词的 synsets。

        Args:
            word: 要获取 synsets 的单词。
        Returns:
            list: 包含给定单词的 synsets 的列表。
        """
        return wn.synsets(word, lang="eng")

    def get_synset_list(self, word: str) -> list[str]:
        """
        使用 WordNet 库来获取一个单词的同义词集（synsets），并返回这些同义词所构成的列表。

        Args:
            word: 要获取同义词集的单词。
        Returns:
            list: 包含给定单词的同义词集的列表。
        """
        return list({synset.lemmas()[0].name() for synset in self._synsets(word)})

    def is_concrete_noun(self, word: str) -> bool:
        """
        检查给定的单词是否是具体名词。

        Args:
            word: 要检查的单词，假定为名词。
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
            # 判断是否属于具体物体的类别
            is_concrete = synsets[0].lexname() not in maynotby_concrete

            self.is_concrete_noun_cache[word] = is_concrete
            # return str(is_concrete) + '\t' + str(synsets[0].lexname())  # for test
            return is_concrete

    def lemma(self, word: str) -> str:
        """
        词形还原
        """
        return self.lemmatizer.lemmatize(word)

    def word_tokenize(self, text: str) -> list[str]:
        """
        分词
        """
        return nltk.word_tokenize(text)
