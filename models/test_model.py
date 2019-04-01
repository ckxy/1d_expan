from torch.autograd import Variable
import torch.nn.functional as F
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import random
import torch


class TestModel(BaseModel):
    def name(self):
        return 'TestModel'

    def initialize(self, opt):
        assert(not opt.isTrain)
        BaseModel.initialize(self, opt)
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)

        self.netG = networks.define_G(opt.input_nc, opt.output_nc,
                                      opt.ngf, opt.which_model_netG,
                                      'batch', not opt.no_dropout,
                                      self.gpu_ids)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

        which_epoch = opt.which_epoch
        self.load_network(self.netG, 'G', which_epoch)

    def set_input(self, input):
        # we need to use single_dataset mode
        input_A = input['A']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.image_paths = input['A_paths']

    def test(self):
        self.real_A = Variable(self.input_A)
        self.fake_B = self.netG.forward(self.real_A)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

    def stress_test_up(self, step=5, crop_size=64):
        input_size = self.input_A.cpu().shape
        width,height = input_size[3], input_size[2]
        results = []
        self.real_A = Variable(self.input_A, volatile=True)
        rw = random.randint(0, width - crop_size)
        rh = random.randint(0, height - crop_size)
        self.real_A = Variable(self.real_A.data, volatile=True)
        self.real_A = Variable(self.real_A.data[:, :, rh:rh + crop_size, rw:rw + crop_size], volatile=True)
        self.fake_B = self.netG.forward(self.real_A)
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        results.append(('real_{}_A'.format(0), real_A))
        results.append(('fake_{}_B'.format(0), fake_B))
        for i in range(1, step):
            # rw = random.randint(0, width)
            # rh = random.randint(0, height)
            # rw = int(width/2)
            # rh = int(height/2)
            self.real_A = Variable(self.fake_B.data, volatile=True)
            print(self.real_A.size())
            self.fake_B = self.netG.forward(self.real_A)
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            results.append(('real_{}_A'.format(i), real_A))
            results.append(('fake_{}_B'.format(i), fake_B))
        return OrderedDict(results)

