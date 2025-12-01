import pandas as pd
import matplotlib.pyplot as plt

class PCSHOW:
    def __init__(self,data:pd.DataFrame,anno:pd.DataFrame=None):
        self.data = data
        self.anno = anno
        pass
    def pcplot(self,x:str, y:str, anno:pd.DataFrame=None,group:str=None,group_order:list=None,color:dict=None,size:dict=None,alpha:dict=None,marker:dict=None,ax:plt.Axes=None,**kwargs):
        anno = self.anno if anno is None else anno
        ax = ax if ax is not None else plt.gca()
        if self.anno is not None or group is not None:
            data = pd.concat([self.data,self.anno[group]],axis=1).dropna()
            groups = self.anno[group].unique() if group_order is None else group_order
            size = dict(zip(groups,[4 for i in groups])) if size is None else size # scatter size
            alpha = dict(zip(groups,[1 for i in groups])) if alpha is None else alpha # scatter size
            color = dict(zip(groups,['blue' for i in groups])) if color is None else color # scatter size
            marker = dict(zip(groups,['o' for i in groups])) if marker is None else marker # scatter size
            for g in groups:
                data_ = data[data[group]==g]
                ax.scatter(x=data_[x],y=data_[y],s=size[g],alpha=alpha[g],marker=marker[g],color=color[g],label=g,**kwargs)
        else:
            data = self.data
            ax.scatter(x=data[x],y=data[y],**kwargs)
        return
    def text_anno(self,x:str, y:str, anno:pd.DataFrame=None,anno_tag:str=None,ax:plt.Axes=None,**kwargs):
        from adjustText import adjust_text
        anno = self.anno if anno is None else anno
        ax = ax if ax is not None else plt.gca()
        data = pd.concat([self.data,anno[anno_tag]],axis=1).dropna()
        texts = []
        for ind,i in enumerate(data.index):
            texts.append(ax.text(data.loc[i][x],data.loc[i][y],data.loc[i][anno_tag],**kwargs))
        adjust_text(
            texts, 
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.7, shrinkA=5),
            expand_points=(1.2, 1.5),
            expand_text=(1.2, 1.5),
            force_text=0.5,
            force_points=0.5,
            precision=0.01
            )
