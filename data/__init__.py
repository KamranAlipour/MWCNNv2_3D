from importlib import import_module

from dataloader import MSDataLoader
from torch.utils.data.dataloader import default_collate
print('Data Class Init.')
class Data:
    def __init__(self, args):
        print('Initializing Data Class...')
        kwargs = {}
        if not args.cpu:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = True
        else:
            kwargs['collate_fn'] = default_collate
            kwargs['pin_memory'] = False

        self.loader_train = None
        if not args.test_only:
            print('Importing data module...')
            module_train = import_module('data.' + args.data_train.lower())
            print('Data module '+args.data_train.lower()+' imported.')
            trainset = getattr(module_train, args.data_train)(args)
            print('Importing data module... : trainset acquired')
            self.loader_train = MSDataLoader(
                args,
                trainset,
                batch_size=args.batch_size,
                shuffle=True,
                **kwargs
            )
        print ('dataset: '+args.data_test)
        if args.data_test in ['Set5', 'Set14', 'B100', 'Urban100']:
            if not args.benchmark_noise:
                module_test = import_module('data.benchmark')
                testset = getattr(module_test, 'Benchmark')(args, train=False)
            else:
                module_test = import_module('data.benchmark_noise')
                testset = getattr(module_test, 'BenchmarkNoise')(
                    args,
                    train=False
                )
        else:
            print('Importing data module...')
            module_test = import_module('data.' +  args.data_test.lower())
            print('Data module '+args.data_train.lower()+' imported.')
            testset = getattr(module_test, args.data_test)(args, train=False)
            print('Importing data module... : testset acquired')
        self.loader_test = MSDataLoader(
            args,
            testset,
            batch_size=1,
            shuffle=False,
            **kwargs
        )
