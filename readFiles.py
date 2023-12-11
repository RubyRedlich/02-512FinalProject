# takes in fasta file, discards first line, outputs gene sequence. 
# fileToGene(filename, True) --> outputs it in all lowercase
# fileToGene(filename, False) --> outputs exons in uppercase, introns in lowercase

def fileToGene(file, lowercase):
    gene = ""
    with open(file, "r") as genefile: 
        info_line = genefile.readline()
        for line in genefile: 
            gene += line.strip()
    if lowercase: 
        gene = gene.lower()
    return gene 

# print(fileToGene("INS", False))