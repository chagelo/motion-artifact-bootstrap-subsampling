from Options.base_options import BaseOptions

class TrainOptions(BaseOptions):
    """
    This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        # parser = BaseOptions.initialize(parser)
        parser =  super().initialize(parser)
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')

        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', default=False, type=bool, help='whether saves model by iteration, if False, then only save the latest model, else save by iteration.')
        parser.add_argument('--save_loss_freq', type=int, default=500, help='every 500 batch save loss once')
        parser.add_argument('--continue_train', default=False, type=bool, help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        
        # training parameters
        # total training epoch = n_epochs + n_epochs_decay
        parser.add_argument('--n_epochs', type=int, default=20, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=30, help='number of epochs to linearly decay learning rate to zero')

        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.00001, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        parser.set_defaults(dataset_mode='mmbs')
        parser.set_defaults(model='mmbs')
        parser.set_defaults(shuffle=True)

        self.isTrain = True
        return parser