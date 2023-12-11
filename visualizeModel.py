import matplotlib.pyplot as plt 
from fractionCorrect2 import fractionCorrect2

def visualize(states, seqstr, gene):
    exonColor = '#19a8af'
    intronColor = '#046e46'

    sawI = False
    intron_regions = []
    exon_regions = []
    if states[0] == 'I':
        sawI = True
        intron_regions.append(0)
    else:
        exon_regions.append(0)

    for i in range(1, len(states)):
        curr = states[i]
        if sawI and curr != 'I':
            sawI = False
            intron_regions.append(i-1)
            exon_regions.append(i-1)
        elif not sawI and curr == 'I':
            sawI = True
            intron_regions.append(i-1)
            exon_regions.append(i-1)

    if len(intron_regions) % 2 == 1:
        intron_regions.append(len(states) - 1)
    elif len(exon_regions) % 2 == 1:
        exon_regions.append(len(states) - 1)

    sawI = False
    seq_intron_regions = []
    seq_exon_regions = []
    if seqstr[0].islower():
        sawI = True
        seq_intron_regions.append(0)
    else:
        seq_exon_regions.append(0)

    for i in range(1, len(seqstr)):
        curr = seqstr[i]
        if sawI and curr.isupper():
            sawI = False
            seq_intron_regions.append(i-1)
            seq_exon_regions.append(i-1)
        elif not sawI and curr.islower():
            sawI = True
            seq_intron_regions.append(i-1)
            seq_exon_regions.append(i-1)

    if len(seq_intron_regions) % 2 == 1:
        seq_intron_regions.append(len(states) - 1)
    elif len(seq_exon_regions) % 2 == 1:
        seq_exon_regions.append(len(states) - 1)

    fig,ax = plt.subplots(figsize=(8,3))

    for i in range(0, len(intron_regions) - 1, 2):
        start = intron_regions[i]
        end = intron_regions[i+1]
        ax.add_patch(plt.Rectangle((start, 1.05), end - start, 0.9, facecolor=intronColor))
    for i in range(0, len(exon_regions) - 1, 2):
        start = exon_regions[i]
        end = exon_regions[i+1]
        ax.add_patch(plt.Rectangle((start, 1), end - start, 1, facecolor=exonColor))
    for i in range(0, len(seq_intron_regions) - 1, 2):
        start = seq_intron_regions[i]
        end = seq_intron_regions[i+1]
        ax.add_patch(plt.Rectangle((start, 3.05), end - start, 0.9, facecolor=intronColor))
    for i in range(0, len(seq_exon_regions) - 1, 2):
        start = seq_exon_regions[i]
        end = seq_exon_regions[i+1]
        ax.add_patch(plt.Rectangle((start, 3), end - start, 1, facecolor=exonColor))

    ax.set_xlim(0, len(states)-1)
    ax.set_ylim(0, 5)
    ax.set_xlabel("Base Pair")
    ax.yaxis.set_visible(False)

    legend_elements = [
        plt.Line2D([0], [0], marker='s', color='w', label='Exon', markersize=10, markerfacecolor=exonColor),
        plt.Line2D([0], [0], marker='s', color='w', label='Intron', markersize=10, markerfacecolor=intronColor)
    ]
    ax.legend(handles=legend_elements)

    plt.title(f"Actual vs Predicted Exon and Intron Regions for {gene}")
    plt.tight_layout()
    plt.show()
    print(fractionCorrect2(states, seqstr))
