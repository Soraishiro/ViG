import weakref
from tqdm import tqdm
from .hooks import HookBase

class SolverBase:

    def __init__(self, model, dataloader, optimizers, device='cuda', lr_scheduler=None):
        self.model = model
        self.dataloader = dataloader
        self.optimizers = optimizers
        if isinstance(self.optimizers, list):
            self.optimizer = optimizers[0]
        self.scheduler = lr_scheduler
        self.hooks = []
        self.step_res = {}
        self.epoch_res = {}
        self.device = device
        self.step = 0
        self.epoch = 0
        self.progbar = None
        self.keys = {'epoch'}

    def register_hooks(self, hooks):
        for h in hooks:
            assert isinstance(h, HookBase)
            h.register(weakref.proxy(self))
        self.hooks.extend(hooks)
        self.hook_name2idx = {h.__class__.__name__: idx for idx, h in enumerate(self.hooks)}

    def exec(self, fn_name):
        for h in self.hooks:
            getattr(h, fn_name)()

    def on_step(self, batch):
        self.model.train()
        if isinstance(self.optimizers, list):
            for optimizer in self.optimizers:
                optimizer.zero_grad()
        else:
            self.optimizers.zero_grad()
        self.step_res = self.model(batch)
        for key, value in self.step_res.items():
            self.epoch_res[key].append(value)
        self.step_res['loss'].backward()
        if isinstance(self.optimizers, list):
            for optimizer in self.optimizers:
                optimizer.step()
        else:
            self.optimizers.step()

    def run_epoch(self, epoch):
        self.epoch = epoch
        self.exec('before_epoch')
        self.progbar = tqdm(self.dataloader)
        for step, batch in enumerate(self.progbar):
            self.step = step
            self.exec('before_step')
            self.on_step(batch)
            self.exec('after_step')
        for key in self.epoch_res:
            self.epoch_res[key] = float(self.epoch_res[key]) / float(len(self.dataloader))
        self.exec('after_epoch')
        for key in self.epoch_res:
            self.epoch_res[key] = 0.0
