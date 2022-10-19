import numpy as np
from Models.base_model import BaseModel
import network

class TestModel(BaseModel):
    """
    This model is used to construct testing.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        Add new test model specific options

        Parameters:
            parser:     original parser
            is_train (bool)
        
        Return:
            modified parser.
        """
        assert not is_train, 'TestModel cannot be used during training time'
        parser.set_defaults(dataset_mode='single')
        parser.add_argument('--model_suffix', type=str, default='', help='In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.')

    def __init__(self, opt):
        """
        Initialize the test model

        Parameters:
            opt (Option class)      store all the experiment flags;
        """
        assert not opt.isTrain
        BaseModel.__init__(self, opt)
        self.loss_names = ['L1']
        # specify the model we want to load, when do test, we we call <BaseModel.load_networks>
        self.model_names = ['net']
        self.net = network.define_net(opt.input_nc, opt.output_nc, opt.net_name, opt.norm, not opt.no_dropout, opt.custom_init, opt.init_type, opt.init_gain, self.gpu_ids)
    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        self.image = input['image'].to(self.device) 
        self.image_paths = input['save_path']
    
    def save_images(self):
        """
        run in test
        """
        n, c, w, h = self.denoised_image.shape

        for i in range(n):
            image = self.denoised_image[i, :, :, :].cpu().detach().numpy()
            path = self.image_paths[i]
            np.save(path, image)


    def forward(self):
        """
        forward
        """
        self.denoised_image = self.net(self.image)

    def optimize_parameters(self):
        """
        for test, do nothing except only forward
        """
        pass
