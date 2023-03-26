"""
@author: sjl
@purpose: preprocess DNase file download from ENCODE with 'bed narrowPeak' format.
"""

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import glob, re
from parameters import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from utils import TimeStatistic

# @TimeStatistic
def make_DNase_statistics_for_one(file_path, draw=False):
    '''
    @purpose: get mean max min of open chromatin region len & signalValue in the narrowPeak bed
    @return: ocr_list, signalValue_list
    '''
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()

    ocr_list = list()
    signalValue_list = list()
    for i in lines:
        # a line data is like
        # chrom chromStart  chromEnd    name    score   strand  signalValue pValue  qValue  peak 
        # chr1  268011      268100      .       0       .       0.497607    -1      -1      75
        # the bed narrowPeak from ENCODE 
        fields = i.strip().split()
        templen = int(fields[2])-int(fields[1])
        ocr_list.append(templen)
        signalValue = float(fields[6])
        signalValue_list.append(signalValue)
        assert fields[3]==".", "name != ."
        assert int(fields[4])==0, "score != 0"
        assert fields[5]==".", "strand != ."
        assert int(fields[7])==-1, "pValue != -1"
        assert int(fields[8])==-1, "qValue != -1"
        assert int(fields[9])==75, "peak != 75"
    print(f"len: min={min(ocr_list)}, max={max(ocr_list)}, mean={np.mean(ocr_list)}, median={np.median(ocr_list)}")
    print(f"signalValue: min={min(signalValue_list)}, max={max(signalValue_list)}, mean={np.mean(signalValue_list)}, median={np.median(signalValue_list)}")
    if draw:
        FONTSIZE = 8
        plt.figure(dpi=300, figsize=(4, 3))
        plt.rc("font", family="Times New Roman")
        params = {"axes.titlesize": FONTSIZE,
                "legend.fontsize": FONTSIZE,
                "axes.labelsize": FONTSIZE,
                "xtick.labelsize": FONTSIZE,
                "ytick.labelsize": FONTSIZE,
                "figure.titlesize": FONTSIZE,
                "font.size": FONTSIZE}
        plt.rcParams.update(params)
        ax = plt.subplot(1, 2, 1)
        # plt.violinplot(ocr_list, showmeans=True, showmedians=True, showextrema=True)
        sns.violinplot(ocr_list)
        plt.xlabel('ocr len')
        # plt.ylabel('True Positive Rate (TPR)')
        plt.title('The statistics for DNase')
        # plt.legend(loc="best")
        ax = plt.subplot(1, 2, 2)
        # plt.violinplot(signalValue_list, showmeans=True, showmedians=True, showextrema=True)
        sns.violinplot(signalValue_list)
        plt.xlabel('signalValue len')
        plt.title('The statistics for DNase')
        # plt.show()
        plt.savefig(fname="dnase_statistics.png", format="png", bbox_inches="tight")
    return ocr_list, signalValue_list

@TimeStatistic
def make_DNase_statistics_for_many(dnase_accessions):
    '''
    @purpose: draw all cell line 的 ocr len & signalValue in a figure
    @parameters: dnase_accessions: a list of dnase_accessions [(cell_line, accession), ...]
    '''
    dnase_accessions = np.array(dnase_accessions)
    cell_line2order = {cell_line:i for i, cell_line in enumerate(dnase_accessions[:, 0])}
    # print(cell_line2order)
    cellLine2ocrLen_list = list()
    cellLine2signalValue_list = list()
    for cell_line, accession in dnase_accessions:
        print(f"Processing {cell_line}, {accession}")
        ocr_list, signalValue_list = make_DNase_statistics_for_one(file_path=ROOT_DIR+f"/data/DNase/{cell_line}/{accession}.bed")
        assert len(ocr_list) == len(signalValue_list)
        cellLine2ocrLen_list.extend([[cell_line2order[cell_line], i] for i in ocr_list])
        cellLine2signalValue_list.extend([[cell_line2order[cell_line], i] for i in signalValue_list])
    cellLine2ocrLen_list = pd.DataFrame(cellLine2ocrLen_list, columns=["cell_line", "ocr len"])
    cellLine2signalValue_list = pd.DataFrame(cellLine2signalValue_list, columns=["cell_line", "signalValue"])

    FONTSIZE = 8
    plt.figure(dpi=300, figsize=(16, 8))
    plt.rc("font", family="Times New Roman")
    params = {"axes.titlesize": FONTSIZE,
            "legend.fontsize": FONTSIZE,
            "axes.labelsize": FONTSIZE,
            "xtick.labelsize": FONTSIZE,
            "ytick.labelsize": FONTSIZE,
            "figure.titlesize": FONTSIZE,
            "font.size": FONTSIZE}
    plt.rcParams.update(params)
    ax = plt.subplot(2, 1, 1)
    sns.violinplot(x="cell_line", y="ocr len", data=cellLine2ocrLen_list, scale="width")
    plt.axhline(100, ls="--", color="grey", lw=0.8)
    # the cell line from glob have already sorted, the sort here is just for secure
    cell_line_order_list = np.array(sorted([[i, cell_line2order[i]] for i in cell_line2order], key=lambda x: x[1]))
    plt.xticks(range(len(cell_line_order_list)), ["" for _ in range(len(cell_line_order_list))], fontsize=FONTSIZE)
    plt.ylim(0, 800)
    plt.xlabel('')
    plt.title('The statistics for DNase')
    ax = plt.subplot(2, 1, 2)
    sns.violinplot(x="cell_line", y="signalValue", data=cellLine2signalValue_list, scale="width")
    plt.axhline(0.35, ls="--", color="grey", lw=0.8)
    plt.xticks(range(len(cell_line_order_list)), cell_line_order_list[:, 0].tolist(), fontsize=FONTSIZE, rotation=315)
    plt.ylim(0, 10)
    # plt.xlabel('signalValue')
    plt.tight_layout()
    # plt.show()
    plt.savefig(fname=ROOT_DIR+r"/figure/dnase_statistics.svg", format="svg", bbox_inches="tight")
    plt.savefig(fname=ROOT_DIR+r"/figure/dnase_statistics.png", format="png", bbox_inches="tight")
    plt.savefig(fname=ROOT_DIR+r"/figure/dnase_statistics.eps", format="eps", bbox_inches="tight")

@TimeStatistic
def filter_narrowPeak_by_length_and_signalValue(dnase_accessions, ocr_length_threshold=100, signalValue_threshold=0.35):
    '''
    @purpose: use open chromatin region length & signalValue to filter high confidence peak,
              make peak region length >= ocr_length_threshold & peak signalValue >= signalValue_threshold in output file
    '''
    for cell_line, accession in dnase_accessions:
        f = open(DNASE_DIR+f"/{cell_line}/{accession}.bed", 'r')
        lines = f.readlines()
        f.close()
        f_out = open(DNASE_DIR+f"/{cell_line}/{accession}.filtered.bed", 'w')
        for i in lines:
            # chrom chromStart  chromEnd    name    score   strand  signalValue pValue  qValue  peak 
            fields = i.strip().split()
            temp_len = int(fields[2])-int(fields[1])
            temp_signalValue = float(fields[6])
            if temp_len < ocr_length_threshold or temp_signalValue < signalValue_threshold: continue
            f_out.write(i)
        f_out.close()


if __name__=="__main__":
    # make_DNase_statistics_for_one(file_path=ROOT_DIR+r"/data/DNase/GM12878/ENCFF759OLD.bed", draw=True)

    dnase_accessions = glob.glob(DNASE_DIR+r"/*/*.bed")
    dnase_accessions = [re.split(r"/|\.", repr(i).replace(r"\\", r'/'))[-3:-1] for i in dnase_accessions] # split by / and .

    # make_DNase_statistics_for_many(dnase_accessions)

    filter_narrowPeak_by_length_and_signalValue(dnase_accessions)

