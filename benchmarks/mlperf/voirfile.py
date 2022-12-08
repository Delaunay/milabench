



def instrument_probe(ov):
    # (
    # ov.probe("/train/main > #endloop__ as step")
    #     .augment(batch=lambda self: self.example_inputs["input_ids"])
    #     .augment(batch_size=lambda batch: len(batch))
    #     .give()
    # )
       
    # (
    # ov.probe("/train/main > train_loss")
    #     .take_last(1)["loss"]
    #     .map(float)
    #     .give("loss")
    # )

    (
        ov.probe("/seq2seq.train.trainer/feed_data > #endloop__ as step")
        .augment(batch=lambda self: self.example_inputs["src"])
        .augment(batch_size=lambda batch: len(batch))
        .give()
    )
    
    (
        ov.probe("/seq2seq.train.trainer/feed_data > losses_per_token")
        .take_last(1)["loss"]
        .map(float)
        .give("loss")
    )