def _strip_wrappers(name):
    while name.startswith('module.'):
        name = name[len('module.'):]
    return name

def is_ext_enabled(config):
    model_ext = getattr(config, 'model_ext', None)
    if model_ext is None:
        return False
    rrm = getattr(model_ext, 'rrm', None)
    lcqm = getattr(model_ext, 'lcqm', None)
    return bool(getattr(rrm, 'enabled', False) or getattr(lcqm, 'enabled', False))

def is_ext_param(name):
    name = _strip_wrappers(name)
    parts = name.split('.')
    if not parts:
        return False
    if parts[0] in {'rrm', 'lcqm'}:
        return True
    ext_leaf_names = {'vis_att3', 'fc_alpha3', 'beta_rel', 'vis_att4', 'fc_alpha4', 'beta_loc', 'eta_loc', 'fusion_ln'}
    return any((part in ext_leaf_names for part in parts))

def trainable_ext_param_names(model):
    model = getattr(model, 'module', model)
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad and is_ext_param(name)]
