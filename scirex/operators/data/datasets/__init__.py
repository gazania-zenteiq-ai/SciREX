from .car_cfd_dataset_jax import CarCFDDatasetjax


def get_dataset(name, config):
    if name == "car_cfd":
        return CarCFDDatasetjax(
            root_dir=config.data.root,
            query_res=[config.data.sdf_query_resolution] * 3,
            n_train=config.data.n_train,
            n_test=config.data.n_test,
            download=config.data.download,
        )
    else:
        raise ValueError(f"Unknown dataset: {name}")