from models.skip_grams import SkipGrams

__all__ = [
    'SkipGrams'
]


def build_model(config, *args, **kwargs):
    if config.model_name in __all__:
        return globals()[config.model_name](config, *args, **kwargs)
    else:
        raise Exception('The model name %s does not exist' % config.model_name)


def get_model_class(config):
    if config.model_name in __all__:
        return globals()[config.model_name]
    else:
        raise Exception('The model name %s does not exist' % config.model_name)
