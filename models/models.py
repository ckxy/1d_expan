
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'half_style_multi_resnet':
        # assert (opt.dataset_mode == 'half_crop')
        from .half_gan_style_mr import HalfGanStyleMRModel
        model = HalfGanStyleMRModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
