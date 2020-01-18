class Config:
    def __init__(self, args):
        # BERT base model num_layers=12, d_model=768, num_heads=12

        self.d_model = args.d_model                   # model vector dimension size
        self.dff = 4 * self.d_model          # feed-forward dense layer size (4*d_model)
        self.num_layers = args.num_layers    
        self.num_heads = args.num_heads
        self.dropout_rate = args.dropout_rate
        self.warmup_steps = args.warmup_steps
        self.target_sentence_max_len = 100
        if args != None:
            self.subwords_file = args.vocab_path
            self.train_ds_file = args.train_file
            self.dev_ds_file = args.dev_file
            self.epoch = args.epoch
            self.batch_size = args.batch_size

            self.checkpoint_dir = args.checkpoint_dir
            self.summary_dir = args.summary_dir
            self.restore = args.restore
