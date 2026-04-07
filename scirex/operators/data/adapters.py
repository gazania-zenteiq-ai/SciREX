def gino_batch_adapter(sample):
    """
    Convert CarCFD sample → SciREX training batch
    """

    # VERY SIMPLE mapping for now (we refine later)
    x = sample["vertices"]        # input
    y = sample["press"].squeeze() #arget

    return {
        "x": x,
        "y": y
    }