import pandas as pd

class GENOMETOOL:
    def __init__(self,genomePath:str):
        print(f'Adjusting ref alt alleles by {genomePath}...')
        chrom = []
        seqs = []
        with open(genomePath,'r') as f:
            for line in f:
                line = line.strip()
                if '>' in line:
                    if len(chrom) > 0:
                        seqs.append(seq)
                    chrom.append(line.split(' ')[0].replace('>',''))
                    seq = []
                else:
                    seq.append(line)
            seqs.append(seq)
        self.genome = dict(zip(chrom,seqs))
    def _readLoc(self,chr,loc):
        strperline = len(self.genome[f'{int(chr)}'][0])
        line = int(loc)//strperline
        strnum = int(loc)%strperline-1
        return self.genome[f'{chr}'][line][strnum]
    def refalt_adjust(self, ref_alt:pd.Series):
        ref_alt = ref_alt.astype(str)
        ref = pd.Series([self._readLoc(i,j) for i,j in ref_alt.index],index=ref_alt.index,name='REF')
        alt:pd.Series = ref_alt.iloc[:,0]*(ref_alt.iloc[:,0]!=ref)+ref_alt.iloc[:,1]*(ref_alt.iloc[:,1]!=ref)
        alt.name = 'ALT'
        ref_alt = pd.concat([ref,alt],axis=1)
        self.error = ref_alt.loc[alt.str.len()!=1,:].index
        ref_alt.loc[self.error,:] = pd.NA
        print(f'Number of sites differ with reference is {len(self.error)}, ratio is {round(len(self.error)/ref_alt.shape[0],3)}')
        self.exchange_loc:bool = (ref_alt.iloc[:,0]!=ref)
        return ref_alt.astype('category')