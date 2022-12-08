import subprocess

from milabench.pack import Package

REPOSITORY = "https://github.com/mlcommons/training.git"
BRANCH = "v0.5"
BENCHNAME = "rnn_translator"


class MLPerfPack(Package):
    @property
    def mlperf_bench(self):
        return self.config["benchmark"]
    
    @property
    def dataset_directory(self):
        return self.dirs.data / "dataset"
      
    @property
    def output_directory(self):
        return self.dirs.data / "output"
      
    @property
    def requirement_file(self):
        return self.dirs.code / self.mlperf_bench / "pytorch" / "requirements.txt"
    
    @property
    def download_script(self):
        return self.dirs.code / self.mlperf_bench / "download_dataset.sh"
    
    def install(self):
        install_method = f'install_{self.mlperf_bench}'
        
        if hasattr(self, install_method):
            getattr(self, install_method)(self)
        else:
            raise RuntimeError(f'{self.mlperf_bench} is not implemented')

    def prepare(self):
        prepare_method = f'prepare_{self.mlperf_bench}'
        
        if hasattr(self, prepare_method):
            getattr(self, prepare_method)(self)
        else:
            raise RuntimeError(f'{self.mlperf_bench} is not implemented')
        
    def run(self, args, voirargs, env):
        run_method = f'run_{self.mlperf_bench}'
        
        if hasattr(self, run_method):
            getattr(self, run_method)(self, args, voirargs, env)
        else:
            raise RuntimeError(f'{self.mlperf_bench} is not implemented')
        
    # RNN translator
    # ==============
    
    def install_rnn_translator(self):
        code = self.dirs.code
        code.clone_subtree(REPOSITORY, BRANCH)

        self.pip_install("-r", self.requirement_file())
        
    def prepare_rnn_translator(self):
        subprocess.check_call(["bash", self.download_script, self.dataset_directory])

    def run_rnn_translator(self, args, voirargs, env):
        """Execute the benchmark"""
        multiproc = self.dirs.code / BENCHNAME / "pytorch" / "multiproc.py"
        train = self.dirs.code / BENCHNAME / "pytorch" / "train.py"
        
        required_args = [
            train,
            '--save', self.output_directory,
            '--dataset-dir', self.dataset_directory,
            '--seed', "1",
            '--target-bleu', "21.80",
            '--epochs', "8", 
            '--batch-size', '128'
        ]
    
        return self.launch(multiproc, args=required_args + list(args), voirargs=voirargs, env=env)


__pack__ = MLPerfPack
