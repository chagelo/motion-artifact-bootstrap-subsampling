from ast import parse
from Options.base_options import BaseOptions


class TestOptions(BaseOptions):
    """
    This class includes test options.
    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        # parser = BaseOptions.initialize(parser)
        parser = super().initialize(parser)
        parser.add_argument('--results_dir', type=str, default='./results/', help='results path')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # parser.add_argument('--model', type=str, default='mmbs_test', help='model for test')
        parser.set_defaults(dataset_mode='mmbstest')
        parser.set_defaults(model='mmbstest')
        parser.set_defaults(batch_size=1)
        parser.set_defaults(num_threads=0)
        parser.set_defaults(shuffle=True)
        self.isTrain = False
        return parser