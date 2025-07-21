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
        初始化 Spacy 模型，根据指定的大小加载不同的模型。

        Args:
            model_size: 模型大小，可以是 "sm", "md", "lg" 或 "trf"。
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
        加载 Spacy 模型。
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
        self._noun_pos: list[str] = ["NOUN", "PROPN"]  # Spacy 的词性标记：NOUN 是普通名词，PROPN 是专有名词

    def _load_fastcoref(self):
        """
        加载 fastcoref 模型，用于指代消解。
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
        输入原始的文本，返回指代消解后的文本。执行一次指代消解大约需要 0.5s。

        Args:
            text: 要解析的文本。
            force_list: 如果为 True，返回一个列表，即使只有一个输入文本。
        Returns:
            list[str] | str: 指代消解后的文本
        """

        if not self._loaded_fastcoref:
            self._load_fastcoref()

        if isinstance(text, str):
            text = [text]
        docs: list[Doc] = list(self.nlp.pipe(text, disable=self._resolve_coref_disable, component_cfg=self._coref_cfg))
        resolved_text: list[str] = [doc._.resolved_text for doc in docs]

        return maybe_return_ls(force_list, resolved_text)

    def get_simi(self, text1: str | Doc, text2: str | Doc) -> float:
        """
        计算两个文本的相似度。
        """
        text1_str = text1 if isinstance(text1, str) else text1.text
        text2_str = text2 if isinstance(text2, str) else text2.text

        if (text1_str, text2_str) in self._similarity_cache:
            return self._similarity_cache[(text1_str, text2_str)]

        doc1: Doc = self.process_text(text1) if isinstance(text1, str) else text1
        doc2: Doc = self.process_text(text2) if isinstance(text2, str) else text2
        if doc1 and doc1.vector_norm and doc2 and doc2.vector_norm:
            similarity: float = doc1.similarity(doc2)
        else:
            similarity = 0.0

        self._similarity_cache[(text1_str, text2_str)] = similarity
        self._similarity_cache[(text2_str, text1_str)] = similarity
        return similarity

    def process_text(self, text: list[str] | str) -> list[Doc] | Doc:
        """
        处理文本，返回 Spacy 的 Doc 对象，如果输入是列表，则返回一个 Doc 对象列表。

        Args:
            text: 要处理的文本。
        Returns:
            list[Doc] | Doc: 处理后的文本。
        """
        if isinstance(text, list):  # 并行处理，提高速度
            return list(self.nlp.pipe(text, disable=self._others_disable))
        else:
            return self.nlp(text, disable=self._others_disable)

    def __call__(self, text: list[str] | str) -> list[Doc] | Doc:
        return self.process_text(text)

    def is_noun(self, word: str | Token | Doc) -> bool:
        """
        检查给定的单词是否是名词（如果用于检查句子，句子中只要有一个名词就返回 True）。
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
        从文本中提取名词（需要 pipeline 中包含 parser 组件）。
        """
        # 注：这样会包括名词的修饰部分，包括冠词、形容词等，需要注意
        doc: Doc = self.process_text(text)
        return [noun.text.lower().replace("the", "").strip() for noun in doc.noun_chunks]

    def lemma(self, word: str) -> str:
        """
        输入一个单词或短语，获取最后一个单词的词干，并返回处理过后的整个短语。

        Args:
            word: 要获取词干的单词或短语。
        Returns:
            str: 输入单词或短语，其中最后一个单词被替换为其词干。
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
