"""The conductor pickles the model into every worker's spawn arguments. Qwen3.5 and Qwen3-Omni
leave their tokenizer and processors out of that payload (only the API server's copy processes
prompts and detokenizes), and a copy that does read one loads it on first use."""

from __future__ import annotations

import logging
import pickle

import pytest

pytest.importorskip("transformers")

from test_qwen3_5_video import write_synthetic_checkpoint  # noqa: E402  (sibling helper)


@pytest.fixture
def qwen3_5(tmp_path, monkeypatch):
    """A Qwen3.5 model over a synthetic checkpoint, and the tokenizer loads made after it."""
    from mstar.model.qwen3_5 import qwen3_5_model

    model = qwen3_5_model.Qwen3_5DenseModel(str(write_synthetic_checkpoint(tmp_path / "repo")))
    loads = []
    real = qwen3_5_model.AutoTokenizer.from_pretrained

    def from_pretrained(*args, **kwargs):
        loads.append(args[0])
        return real(*args, **kwargs)

    monkeypatch.setattr(qwen3_5_model.AutoTokenizer, "from_pretrained", from_pretrained)
    return model, loads


def test_qwen3_5_pickles_without_its_tokenizer_and_reloads_it_on_first_use(qwen3_5):
    model, loads = qwen3_5
    payload = pickle.dumps(model)
    assert len(payload) < len(pickle.dumps(model.tokenizer))
    worker = pickle.loads(payload)
    assert "tokenizer" not in vars(worker) and loads == []
    # what the workers read survived: stop ids, the media pads, the configs
    assert worker.config.stop_token_ids == model.config.stop_token_ids
    assert [(m.modality, m.pad) for m in worker.media] == [(m.modality, m.pad) for m in model.media]
    # a copy that tokenizes loads it once, from the same checkpoint
    text = "<|im_start|>user\nhello<|im_end|>"
    assert worker.tokenizer.encode(text) == model.tokenizer.encode(text)
    assert worker.tokenizer is worker.tokenizer and loads == [model.local_dir]
    # and the original keeps its own
    assert "tokenizer" in vars(model)


def test_qwen3_5_other_missing_attributes_still_raise(qwen3_5):
    model, loads = qwen3_5
    worker = pickle.loads(pickle.dumps(model))
    with pytest.raises(AttributeError, match="no_such_field"):
        worker.no_such_field  # noqa: B018
    # a bare instance (as tests build one) has no checkpoint to load from
    from mstar.model.qwen3_5.qwen3_5_model import Qwen3_5DenseModel

    with pytest.raises(AttributeError, match="tokenizer"):
        object.__new__(Qwen3_5DenseModel).tokenizer  # noqa: B018
    assert loads == []


class _Big:
    """A stand-in for a tokenizer or processor: megabytes when pickled."""

    def __init__(self, tag: str):
        self.tag, self.blob = tag, b"x" * (4 << 20)


@pytest.mark.parametrize("processor_loads", [True, False], ids=["processor", "no-processor"])
def test_qwen3_omni_pickles_without_its_tokenizer_and_processor(monkeypatch, caplog, processor_loads):
    import transformers

    from mstar.model.qwen3_omni import qwen3_omni_model
    from mstar.model.qwen3_omni.qwen3_omni_model import Qwen3OmniModel

    loads = []

    def tokenizer(path, **kwargs):
        loads.append(("tokenizer", path, kwargs["trust_remote_code"]))
        return _Big("tokenizer")

    def processor(path, **kwargs):
        loads.append(("processor", path, kwargs["trust_remote_code"]))
        if not processor_loads:
            raise OSError("no processor files")
        return _Big("processor")

    monkeypatch.setattr(qwen3_omni_model.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", processor)
    model = object.__new__(Qwen3OmniModel)
    model.local_dir, model.cache_dir = "/ckpt/omni", None
    model.tokenizer, model._processor = model._load_tokenizer(), model._load_processor()
    model._submodule_cache = {}
    loads.clear()

    payload = pickle.dumps(model)
    assert len(payload) < 64 << 10
    worker = pickle.loads(payload)
    assert "tokenizer" not in vars(worker) and "_processor" not in vars(worker) and loads == []
    assert worker.tokenizer.tag == "tokenizer"
    with caplog.at_level(logging.WARNING):
        expected = "processor" if processor_loads else None
        assert getattr(worker._processor, "tag", None) == expected
        assert worker._processor is worker._processor
    assert loads == [("tokenizer", "/ckpt/omni", False), ("processor", "/ckpt/omni", False)]
    assert model.tokenizer.tag == "tokenizer"
    with pytest.raises(AttributeError, match="no_such_field"):
        worker.no_such_field  # noqa: B018
