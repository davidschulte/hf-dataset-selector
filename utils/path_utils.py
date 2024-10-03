import os


def get_output_path(base_dir,
                    optional_layers=None,
                    bottleneck=False,
                    num_epochs=None,
                    num_train_samples=None,
                    num_source_samples=None,
                    seed=None,
                    source_name=None,
                    target_name=None,
                    filename=None):
    # if isinstance(optional_layers, list):
    optional_layers_str = f'layers_{str(optional_layers)}' if isinstance(optional_layers, list) else ''
    if bottleneck:
        optional_layers_str += '_bottleneck'

    return os.path.join(
        base_dir,
        optional_layers_str,
        f'epochs_{num_epochs}' if num_epochs else '',
        f'samples_{num_train_samples}' if num_train_samples else '',
        f'source_samples_{num_source_samples}' if num_source_samples else '',
        f'seed_{seed}' if seed else '',
        target_name if target_name else '',
        source_name if source_name else '',
        filename if filename else ''
    )
