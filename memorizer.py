import matplotlib.pyplot as plt

class MemRec(object):

    def __init__(self):
        super().__init__()
        self.mem_self_cpu = {}
        self.mem_cpu = {}
        self.mem_self_cuda = {}
        self.mem_cuda = {}

    def get_mem_helper(self, ret_list, col_name):
        index = ret_list[0].index(col_name)
        mem_all_layers_list = [lis[index] for lis in ret_list]
        mem_all_layers_list.pop(0)
        mem_all_layers = 0
        for item in mem_all_layers_list:
            mem, unit = item.rstrip().split()
            # target unit is Kb
            mem = float(mem)
            if unit == "b":
                mem *= 0.000001
            elif unit == "Kb":
                mem *= 0.001
            elif unit == "Mb":
                mem *= 1
            elif unit == "Gb":
                mem *= 1000
            else:
                print(unit)
                exit(1)
            mem_all_layers += mem
        return mem_all_layers

    def get_mem(self, layername, prof_report, usingcuda=True):

        if layername not in self.mem_self_cpu:
            self.mem_self_cpu[layername] = []
            self.mem_cpu[layername] = []
            if usingcuda:
                self.mem_self_cuda[layername] = []
                self.mem_cuda[layername] = []

        ret_list = parse_prof_table(prof_report)

        self.mem_self_cpu[layername].append(self.get_mem_helper(ret_list, "Self CPU Mem"))
        self.mem_cpu[layername].append(self.get_mem_helper(ret_list, "CPU Mem"))
        if usingcuda:
            self.mem_self_cuda[layername].append(self.get_mem_helper(ret_list, "Self CUDA Mem"))
            self.mem_cuda[layername].append(self.get_mem_helper(ret_list, "CUDA Mem"))

    def report(self, sample=False):
        if sample:
            print("--------------------------------")
            print("Self CPU Mem")
            for key, value in self.mem_self_cpu.items():
                print(key, end=' :: ')
                print(value, flush=True)
            print("--------------------------------")
            print()
            print("--------------------------------")
            print("CPU Mem")
            for key, value in self.mem_cpu.items():
                print(key, end=' :: ')
                print(value, flush=True)
            print("--------------------------------")
            if self.mem_cuda:
                print()
                print("--------------------------------")
                print("Self CUDA Mem")
                for key, value in self.mem_self_cuda.items():
                    print(key, end=' :: ')
                    print(value, flush=True)
                print("--------------------------------")
                print()
                print("--------------------------------")
                print("CUDA Mem")
                for key, value in self.mem_cuda.items():
                    print(key, end=' :: ')
                    print(value, flush=True)
                print("--------------------------------")
        
        layernames = []
        avg_mems = []
        print("mem_self_cpu | Average Mem Consumption of Each Layer")
        for key, value in self.mem_self_cpu.items():
            value.pop(0)
            print(f"{key} :: {sum(value)/len(value)}")
            layernames.append(key)
            avg_mems.append(sum(value)/len(value))

        fig = plt.figure()
        # creating the bar plot
        plt.bar(layernames, avg_mems, width = 0.4)
        
        plt.xlabel("layernames")
        plt.ylabel("avg_mem_self_cpu")
        plt.savefig("layer-avg_mem_self_cpu.png")

        layernames = []
        avg_mems = []
        print("mem_cpu | Average Mem Consumption of Each Layer")
        for key, value in self.mem_cpu.items():
            value.pop(0)
            print(f"{key} :: {sum(value)/len(value)}")
            layernames.append(key)
            avg_mems.append(sum(value)/len(value))

        fig = plt.figure()
        # creating the bar plot
        plt.bar(layernames, avg_mems, width = 0.4)
        
        plt.xlabel("layernames")
        plt.ylabel("avg_mem_cpu")
        plt.savefig("layer-avg_mem_cpu.png")

        if self.mem_cuda:
            layernames = []
            avg_mems = []
            print("mem_self_cuda | Average Mem Consumption of Each Layer")
            for key, value in self.mem_self_cuda.items():
                value.pop(0)
                print(f"{key} :: {sum(value)/len(value)}")
                layernames.append(key)
                avg_mems.append(sum(value)/len(value))

            fig = plt.figure()
            # creating the bar plot
            plt.bar(layernames, avg_mems, width = 0.4)
            
            plt.xlabel("layernames")
            plt.ylabel("avg_mem_self_cuda")
            plt.savefig("layer-avg_mem_self_cuda.png")

            layernames = []
            avg_mems = []
            print("mem_cuda | Average Mem Consumption of Each Layer")
            for key, value in self.mem_cuda.items():
                value.pop(0)
                print(f"{key} :: {sum(value)/len(value)}")
                layernames.append(key)
                avg_mems.append(sum(value)/len(value))

            fig = plt.figure()
            # creating the bar plot
            plt.bar(layernames, avg_mems, width = 0.4)
            
            plt.xlabel("layernames")
            plt.ylabel("avg_mem_cuda")
            plt.savefig("layer-avg_mem_cuda.png")




def parse_prof_table(prof_report):

    ret_list = []

    flip = False
    parsing_str = prof_report[0]
    parsing_idx = []
    for i in range(len(parsing_str)):
        if parsing_str[i] == '-':
            flip = True
        if flip and parsing_str[i] == ' ':
            parsing_idx.append(i)
            flip = False

    head_str_list = []
    parsing_str = prof_report[1]
    head_str = ""
    for i in range(len(parsing_str)):
        if i-1 in parsing_idx:
            head_str_list.append(head_str.lstrip().rstrip())
            head_str = ""
        else:
            head_str += parsing_str[i:i+1]
    
    ret_list.append(head_str_list)

    parsing_str_list = prof_report[3:-4]
    for parsing_str in parsing_str_list:
        head_str_list = []
        head_str = ""
        for i in range(len(parsing_str)):
            if i-1 in parsing_idx:
                head_str_list.append(head_str.lstrip().rstrip())
                head_str = ""
            else:
                head_str += parsing_str[i:i+1]
        ret_list.append(head_str_list)

    return ret_list
