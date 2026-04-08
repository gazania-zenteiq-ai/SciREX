def gino_batch_adapter(sample):
    return {
        "input_geom": sample["vertices"][None, ...],
        "latent_queries": sample["query_points"][None, ...],
        "output_queries": sample["vertices"][None, ...],
        "x": sample["distance"][None, ...],
        "y": sample["press"].squeeze(),
        "neighbors_in": sample["neighbors_in"],
        "neighbors_out": sample["neighbors_out"],
    }