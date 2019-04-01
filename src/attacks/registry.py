_ATTACKS = dict()


def register(fn):
  global _ATTACKS
  _ATTACKS[fn.__name__] = fn
  return fn


def get_attack(model, device, criterion, model_params, attack_params):
  return _ATTACKS[attack_params.name](model, device, criterion, model_params,
                                      attack_params)