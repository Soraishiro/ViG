import contextlib, sys

class DummyFile(object):

    def write(self, x):
        pass

@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = DummyFile()
    yield
    sys.stdout = save_stdout

def reporthook(t):
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner

def get_tokenizer(tokenizer):
    if callable(tokenizer):
        return tokenizer
    if tokenizer == 'spacy':
        try:
            import spacy
            spacy_en = spacy.load('en_core_web_sm')
            return lambda s: [tok.text for tok in spacy_en.tokenizer(s)]
        except ImportError:
            print('Please install SpaCy and the SpaCy English tokenizer. See the docs at https://spacy.io for more information.')
            raise
        except AttributeError:
            print('Please install SpaCy and the SpaCy English tokenizer. See the docs at https://spacy.io for more information.')
            raise
