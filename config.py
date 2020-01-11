class Config:
    def __init__(self, args, training_mode):
        # BERT base model num_layers=12, d_model=768, num_heads=12
        #Constants
        self.d_model = 256                   # model vector dimension size
        self.dff = 4 * self.d_model          # feed-forward dense layer size (4*d_model)
        self.num_layers = 8    
        self.num_heads = 8
        self.dropout_rate = 0.1
        self.target_sentence_max_len = 100

        self.training_mode = training_mode
        if args != None:
            self.subwords_file = args.vocab_path
            self.train_ds_file = args.train_file
            self.dev_ds_file = args.dev_file
            self.epoch = args.epoch
            self.batch_size = args.batch_size

            self.checkpoint_dir = args.checkpoint_dir
            self.summary_dir = args.summary_dir
            self.restore = args.restore
