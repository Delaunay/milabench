from collections import defaultdict
from milabench.testing import replay_run
from cantilever.core.statstream import StatStream
import os
import json



def extract_from_events():
    # folder = "/home/mila/d/delaunap/overhead_reports/resnet152_dali_async/"
    folder = "/home/mila/d/delaunap/overhead_reports/resnet152_torch_async/"
    folder = "/home/mila/d/delaunap/overhead_reports/resnet152_torch_sync/"
    folder = "/home/mila/d/delaunap/overhead_reports/resnet50_torch_rs_async/"
    # folder = "/home/mila/d/delaunap/overhead_reports/resnet50_torch_rs_sync/"

    def replay_multi_runs(folder):
        for run in os.scandir(folder):
            yield from replay_run(os.path.join(folder, run))


    data = defaultdict(lambda: StatStream(0))

    for entry in replay_multi_runs(folder):
        if entry.event == "line" and entry.data.startswith("#"):

            name, avg, total, sd, count = entry.data.split("|")
            
            name = name.replace("#", "").replace(".", "").replace("_", "").strip()
            
            if name != "":
                data[name].update(float(avg), 1)


    # [self.avg, self.sd, self.min, self.max, self.count]

    header = ["avg", "sd", "min", "max", "count"]
    print(" | ".join([f"{'metric':>10}"] +[f"{v:>10}" for v in header]))
    for k, v in data.items():
        values = v.to_array()
        print(" | ".join([f"{k:>10}"] + [f"{v:10.2f}" for v in values]))
        


def extract_from_stdout():
    # folder = "/home/mila/d/delaunap/overhead_reports/novoir/resnet152_pytorch/"
    folder = "/home/mila/d/delaunap/overhead_reports/novoir/resnet152_pytorch/"
    folder = "/home/mila/d/delaunap/overhead_reports/novoir/resnet152_dali/"
    folder = "/home/mila/d/delaunap/overhead_reports/novoir/resnet50_pytorch/"

    data = defaultdict(lambda: StatStream(0))
    
    for run in os.scandir(folder):
        for line in open(run, "r").readlines():
            if line.startswith("#"):
                name, avg, total, sd, count = line.split("|")
                
                name = name.replace("#", "").replace(".", "").replace("_", "").strip()
                
                if name != "":
                    data[name].update(float(avg), int(1))

    header = ["avg", "sd", "min", "max", "count"]
    print(" | ".join([f"{'metric':>10}"] +[f"{v:>10}" for v in header]))
    for k, v in data.items():
        values = v.to_array()
        print(" | ".join([f"{k:>10}"] + [f"{v}" for v in values]))
        

# extract_from_stdout()
extract_from_events()