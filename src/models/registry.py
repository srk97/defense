_MODELS = dict()


def register(fn):
  global _MODELS
  _MODELS[fn.__name__] = fn()
  return fn


def get_model(name=None):
  if name is None:
    return _MODELS
  return _MODELS[name]