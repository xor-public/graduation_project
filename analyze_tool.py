import argparse
import matplotlib.pyplot as plt
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("logfile",type=str)
    parser.add_argument("-a","--attack",type=str)
    parser.add_argument("-d","--defend",type=str)
    args=parser.parse_args()
    task=text_search(args.logfile,"task")
    plt.title(task.split(" ")[-1])
    plot_accs(args.logfile)
    plt.legend(["baseline"])
    if args.attack:
        plot_accs(args.attack,"r")
        plt.legend(["baseline","attack"])
    if args.defend:
        plot_accs(args.defend,"g")
        plt.legend(["baseline","attack","defend"])
    plt.show()
def plot_accs(logfile,color="b"):
    with open(logfile) as f:
        lines=f.readlines()
    lines=[line.strip() for line in lines]
    epoches=[line.split(" ")[-1] for line in lines if line.startswith("Epoch")]
    accs=[line.split(" ")[-2].strip("()") for line in lines if line.startswith("Val")]
    epoches=[int(epoch) for epoch in epoches]
    accs=[eval(acc) for acc in accs]
    if len(epoches)>len(accs):
        epoches=epoches[:-1]
    plt.plot(epoches,accs)
def text_search(file,keyword):
    with open(file) as f:
        lines=f.readlines()
    lines=[line.strip() for line in lines]
    for line in lines:
        if line.startswith(keyword):
            return line
if __name__ == "__main__":
    main()
