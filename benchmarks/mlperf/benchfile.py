import subprocess

from milabench.pack import Package

REPOSITORY = "https://github.com/mlcommons/training.git"
BRANCH = "master"


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
    
    def bench_dispatch(self, method, *args, **kwargs):
        method = f'{method}_{self.mlperf_bench}'
        
        if hasattr(self, method):
            getattr(self, method)(*args, **kwargs)
        else:
            raise RuntimeError(f'{method} is not implemented')  
    
    def install(self):
        code = self.dirs.code
        code.clone_subtree(REPOSITORY, BRANCH)
        
        self.bench_dispatch("install")

    def prepare(self):
        self.bench_dispatch("prepare")
        
    def run(self, args, voirargs, env):
        self.bench_dispatch("run")
        
        
    # RNN Speech Recognition
    # ======================
        
    def install_rnn_speech_recognition(self):
        reqfile = self.dirs.code / self.mlperf_bench / "pytorch" / "requirements.txt"
        self.pip_install("-r", reqfile)
        
    def prepare_rnn_speech_recognition(self):
        download_script = self.dirs.code / "rnn_speech_recognition" / "pytorch" / "scripts" / "download_librispeech.sh"
        subprocess.check_call(["bash", download_script])
    
        preprocess = self.dirs.code / "rnn_speech_recognition" / "pytorch" / "scripts" / "preprocess_librispeech.sh"
        subprocess.check_call(["bash", preprocess])
        
    def run_rnn_speech_recognition(self, args, voirargs, env):
        pass
    
    # RNN translator
    # ==============
    
    def install_rnn_translator(self):
        reqfile = self.dirs.code / self.mlperf_bench / "pytorch" / "requirements.txt"
        self.pip_install("-r", reqfile)
        
    def prepare_rnn_translator(self):
        download_script = self.dirs.code / "rnn_translator" / "download_dataset.sh"
        
        subprocess.check_call(["bash", download_script, self.dataset_directory])

    def run_rnn_translator(self, args, voirargs, env):
        """Execute the benchmark"""
        multiproc = self.dirs.code / "rnn_translator" / "pytorch" / "multiproc.py"
        train = self.dirs.code / "rnn_translator" / "pytorch" / "train.py"
        
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
