import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
color_set = ['#E4391B','#398249','#F9992A','#714F91','#9C5E27','#739CCD','black']
marker_set = ['+','x','o','8','s','p','P','D','2','1','^']
class PCSHOW:
    def __init__(self,data:pd.DataFrame,):
        self.data = data
        pass
    def pcplot(self,x:str, y:str, anno:pd.DataFrame=None,group:str=None,group_order:list=None,color:dict=None,size:dict=None,alpha:dict=None,marker:dict=None,ax:plt.Axes=None,**kwargs):
        ax = ax if ax is not None else plt.gca()
        if anno is not None and group is not None:
            anno = anno.fillna('others')
            data = pd.concat([self.data,anno[group]],axis=1).dropna()
            groups = anno[group].unique() if group_order is None else group_order
            size = dict(zip(groups,[32 for i in groups])) if size is None else size
            alpha = dict(zip(groups,[1 for i in groups])) if alpha is None else alpha
            color = dict(zip(groups,color_set[:len(groups)])) if color is None else color
            marker = dict(zip(groups,marker_set[:len(groups)])) if marker is None else marker
            size['others'] = 4
            alpha['others'] = .4
            color['others'] = 'grey'
            marker['others'] = '*'
            for g in groups:
                data_ = data[data[group]==g]
                ax.scatter(x=data_[x],y=data_[y],s=size[g],alpha=alpha[g],marker=marker[g],color=color[g],label=g,**kwargs)
            ax.legend()
        else:
            data = self.data
            ax.scatter(x=data[x],y=data[y],**kwargs)
        return
    def text_anno(self,x:str, y:str, anno:pd.DataFrame=None,anno_tag:str=None,ax:plt.Axes=None,**kwargs):
        from adjustText import adjust_text
        ax = ax if ax is not None else plt.gca()
        data = pd.concat([self.data,anno[anno_tag]],axis=1).dropna()
        texts = []
        for ind,i in enumerate(data.index):
            texts.append(ax.text(data.loc[i][x],data.loc[i][y],data.loc[i][anno_tag],**kwargs))
        adjust_text(texts, 
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.7, shrinkA=5),
            expand_points=(1.2, 1.5),
            expand_text=(1.2, 1.5),
            force_text=0.5,
            force_points=0.5,
            precision=0.01)
