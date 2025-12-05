import os
from logging import WARNING, Logger

from spacy.language import Language
from spacy.tokens import Doc, Token

from ..utils.utils import maybe_return_ls


class SpacyModel:
    def __init__(
        self,
        model_size: str = "md",
        model_dir: str = "",
        device: str = "cuda:0",
        logger: Logger | None = None,
    ):
        """
        Initialize Spacy model and load different models according to the specified size.

        Args:
            model_size: Model size, can be "sm", "md", "lg" or "trf".
        """
        assert model_size in ["sm", "md", "lg", "trf"], "Invalid model size. Choose from 'sm', 'md', 'lg 'or 'trf'."

        if logger:
            logger.info(f"Loading Spacy model with size {model_size}")
        self._init_const()
        self.model_size = model_size
        self.model_dir = model_dir
        self.spacy_dir = os.path.join(os.path.dirname(os.path.expanduser(self.model_dir)), "spacy") if model_dir else ""
        self.device = device
        self.nlp: Language = self._load()
        # Cache
        self._is_noun_cache: dict[str, bool] = {}
        self._lemma_cache: dict[str, str] = {}
        self._similarity_cache: dict[tuple[str, str], float] = {}

    def _load(self) -> Language:
        """
        Load Spacy model.
        """

        import spacy

        spacy.logger.setLevel(WARNING)

        model_name = f"en_core_web_{self.model_size}"
        self._loaded_fastcoref: bool = False
        if "cuda" in self.device:
            spacy.prefer_gpu()

        if self.spacy_dir:
            model_path: str = os.path.join(self.spacy_dir, model_name)
            try:
                nlp = spacy.load(model_path, exclude=self._load_exclude)
            except OSError:
                try:
                    nlp = spacy.load(model_name, exclude=self._load_exclude)
                except OSError:
                    spacy.cli.download(model_name)
                    nlp = spacy.load(model_name, exclude=self._load_exclude)
                    nlp.to_disk(model_path)
        else:
            nlp = spacy.load(model_name, exclude=self._load_exclude)
        return nlp

    def _init_const(self) -> None:
        self._load_exclude = [
            "ner",  # Named Entity Recognition
            "textcat",
            "entity_linker",
            "entity_ruler",
            "textcat_multilabel",
        ]
        self._resolve_coref_disable = [
            "parser",
            "lemmatizer",
            "ner",
            "textcat",
            "entity_linker",
            "entity_ruler",
            "textcat_multilabel",
            "trainable_lemmatizer",
            # "attribute_ruler" is essential
        ]
        self._others_disable = [
            "fastcoref",
            "ner",
            "textcat",
            "entity_linker",
            "entity_ruler",
            "textcat_multilabel",
        ]
        self._coref_cfg = {
            "fastcoref": {"resolve_text": True},
        }
        # Spacy POS tags: NOUN is common noun, PROPN is proper noun
        self._noun_pos: list[str] = ["NOUN", "PROPN"]

    def _load_fastcoref(self):
        """
        Load fastcoref model for coreference resolution.
        """
        from fastcoref.spacy_component import FastCorefResolver  # noqa: F401

        self.nlp.add_pipe(
            "fastcoref",
            config={
                "model_architecture": "LingMessCoref",
                "model_path": "biu-nlp/lingmess-coref",
                "device": self.device,
            },
        )
        self._loaded_fastcoref = True

    def resolve_coref(self, text: list[str] | str, force_list: bool = False) -> list[str] | str:
        """
        Input raw text, return text after coreference resolution. Each resolution takes about 0.5s.

        Args:
            text: Text to be resolved.
            force_list: If True, always return a list even for single input.
        Returns:
            list[str] | str: Text after coreference resolution
        """

        if not self._loaded_fastcoref:
            self._load_fastcoref()

        if isinstance(text, str):
            text = [text]
        
        try:
            docs: list[Doc] = list(self.nlp.pipe(text, disable=self._resolve_coref_disable, component_cfg=self._coref_cfg))
            resolved_text: list[str] = [doc._.resolved_text for doc in docs]
        except (TypeError, AttributeError):
            # fastcoref may fail on certain inputs due to internal bugs, fallback to original text
            resolved_text = list(text)

        return maybe_return_ls(force_list, resolved_text)

    def process_text(self, text: list[str] | str) -> list[Doc] | Doc:
        """
        Process text and return Spacy Doc object. If input is a list, return a list of Doc objects.

        Args:
            text: Text to process.
        Returns:
            list[Doc] | Doc: Processed text.
        """
        if isinstance(text, list):  # 并行处理，提高速度
            return list(self.nlp.pipe(text, disable=self._others_disable))
        else:
            return self.nlp(text, disable=self._others_disable)

    def __call__(self, text: list[str] | str) -> list[Doc] | Doc:
        return self.process_text(text)

    def is_noun(self, word: str | Token | Doc) -> bool:
        """
        Check if the given word is a noun (if checking a sentence, returns True if any word is a noun).
        """
        word_text: str = word.text if isinstance(word, (Token, Doc)) else word

        if word_text in self._is_noun_cache:
            return self._is_noun_cache[word_text]

        if isinstance(word, str):
            is_noun: bool = any(token.pos_ in self._noun_pos for token in self(word))
        elif isinstance(word, Token):
            is_noun: bool = word.pos_ in self._noun_pos
        elif isinstance(word, Doc):
            is_noun: bool = any(token.pos_ in self._noun_pos for token in word)
        else:
            raise ValueError("Unsupported type for word")

        self._is_noun_cache[word_text] = is_noun
        return is_noun

    def extract_nouns_from_text(self, text: str) -> list[str]:
        """
        Extract nouns from text (requires parser component in pipeline).
        """
        # 注：这样会包括名词的修饰部分，包括冠词、形容词等，需要注意
        doc: Doc = self.process_text(text)
        return [noun.text.lower().replace("the", "").strip() for noun in doc.noun_chunks]

    def lemma(self, word: str) -> str:
        """
        Input a word or phrase, get the lemma of the last word, and return the processed phrase.

        Args:
            word: Word or phrase to get lemma for.
        Returns:
            str: Input word or phrase with the last word replaced by its lemma.
        """
        if word in self._lemma_cache:
            return self._lemma_cache[word]
        else:
            doc: Doc = self.process_text(word)
            if len(doc) > 1:
                # 对最后一个单词进行词干提取
                lemma: str = doc[-1].lemma_
                # 将最后一个单词替换为其词干
                processed_word = word.rsplit(" ", 1)[0] + " " + lemma
            else:
                lemma: str = doc[0].lemma_
                processed_word = lemma
            self._lemma_cache[word] = processed_word
            return processed_word
