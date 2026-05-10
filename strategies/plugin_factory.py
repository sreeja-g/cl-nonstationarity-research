from avalanche.training.plugins import (
    EWCPlugin,
    ReplayPlugin,
    LwFPlugin,
    AGEMPlugin,
    GEMPlugin,
)


def build_plugins(plugin_names=None, **kwargs):

    if not plugin_names:
        return []

    plugins = []

    for plugin_name in plugin_names:

        plugin_name = plugin_name.lower().strip()

        if plugin_name == "ewc":

            plugins.append(
                EWCPlugin(
                    ewc_lambda=kwargs.get(
                        "ewc_lambda",
                        0.4,
                    ),

                    mode=kwargs.get(
                        "ewc_mode",
                        "separate",
                    ),

                    decay_factor=kwargs.get(
                        "ewc_decay_factor",
                        None,
                    ),

                    keep_importance_data=kwargs.get(
                        "ewc_keep_importance_data",
                        False,
                    ),
                )
            )

        elif plugin_name == "replay":

            plugins.append(
                ReplayPlugin(
                    mem_size=kwargs.get(
                        "mem_size",
                        200,
                    ),

                    batch_size=kwargs.get(
                        "batch_size",
                        None,
                    ),

                    batch_size_mem=kwargs.get(
                        "batch_size_mem",
                        None,
                    ),

                    task_balanced_dataloader=kwargs.get(
                        "task_balanced_dataloader",
                        False,
                    ),

                    storage_policy=kwargs.get(
                        "storage_policy",
                        None,
                    ),
                )
            )

        elif plugin_name == "lwf":

            plugins.append(
                LwFPlugin(
                    alpha=kwargs.get(
                        "lwf_alpha",
                        1.0,
                    ),

                    temperature=kwargs.get(
                        "lwf_temperature",
                        2.0,
                    ),
                )
            )

        elif plugin_name == "agem":

            plugins.append(
                AGEMPlugin(
                    patterns_per_exp=kwargs.get(
                        "agem_patterns_per_exp",
                        256,
                    ),

                    sample_size=kwargs.get(
                        "agem_sample_size",
                        256,
                    ),
                )
            )

        elif plugin_name == "gem":

            plugins.append(
                GEMPlugin(
                    patterns_per_exp=kwargs.get(
                        "gem_patterns_per_exp",
                        256,
                    ),

                    memory_strength=kwargs.get(
                        "gem_memory_strength",
                        0.5,
                    ),
                )
            )

        else:
            raise ValueError(
                f"Unknown plugin_name: {plugin_name}"
            )

    return plugins