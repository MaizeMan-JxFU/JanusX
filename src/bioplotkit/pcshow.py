import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from bioplotkit.sci_set import marker_set

class PCSHOW:
    def __init__(self,data:pd.DataFrame,):
        self.data = data.replace('others',pd.NA)
        self.color = None
        self.marker = None
        pass
    def pcplot(self,x:str, y:str,group:str=None,group_order:list=None,color_set:list=['red','green'],marker_set:list=marker_set,anno_tag:str=None,ax:plt.Axes=None,**kwargs):
        ax = ax if ax is not None else plt.gca()
        if group is not None:
            groupmask = self.data[group].isna()
            groups = self.data.loc[~groupmask,group].unique() if group_order is None else group_order
            color = dict(zip(groups,[color_set[i%len(color_set)] for i in range(len(groups))]))
            marker = dict(zip(groups,[marker_set[i%len(marker_set)] for i in range(len(groups))]))
            self.color = color
            self.marker = marker
            for g in groups:
                data_ = self.data[self.data[group]==g]
                ax.scatter(x=data_[x],y=data_[y],s=32,alpha=.8,marker=marker[g],color=color[g],label=g,**kwargs)
            if np.sum(groupmask)>0:
                data_ = self.data.loc[groupmask]
                ax.scatter(x=data_[x],y=data_[y],s=8,alpha=.4,marker='*',color='grey',label='others',**kwargs)
            ax.legend()
        else:
            data = self.data
            ax.scatter(x=data[x],y=data[y],**kwargs)
        if anno_tag:
            data = self.data.iloc[:,[0,1,2,4]].dropna()
            if not data.empty:
                for i in data.index:
                    ax.text(data.loc[i,x],data.loc[i,y],data.loc[i,anno_tag],ha='center',**kwargs)
    def pcplot3D(self,x:str,y:str,z:str,group:str=None,anno_tag:str=None,color_set:list=['red','green'],equal_aspect:bool=True):
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            print('It is a beta version for 3Dplot')
        except:
            raise ImportError("Please use pip install plotly or conda install plotly")
        data = self.data.reset_index().copy()
        data['size'] = 8
        if group:
            data[group] = data[group].fillna('others')
            groups = data[group].unique()
            color = dict(zip(groups,[color_set[i%len(color_set)] for i in range(len(groups))]))
            color['others'] = '#DCDCDC'
            data.loc[data[group]=='others','size'] = 1
        fig = px.scatter_3d(data,
                            x=x,
                            y=y,
                            z=z,
                            hover_name=data.columns[0],
                            color='group' if group is not None else None,
                            size='size',
                            color_discrete_map=color if group is not None else None,)
        if group and anno_tag:
            data_anno = data.loc[~data[anno_tag].isna()]
            if not data_anno.empty:
                fig.add_trace(go.Scatter3d(
                    x=data_anno[x],
                    y=data_anno[y],
                    z=data_anno[z],
                    mode='markers+text',  # show markers and text
                    marker=dict(
                        symbol='circle-open',
                        color='#6ff01a',
                        size=8,
                        opacity=1.0,
                        line=dict(width=2, color='#6ff01a')
                    ),
                    text=data_anno[anno_tag],  # show id text
                    textposition="top center",  # text loc
                    textfont=dict(
                        size=12,
                        color='black'
                    ),
                    name='Hilighted',
                    hoverinfo='text',
                    showlegend=True
                ))
        fig.update_layout(
        title='PCA',
        scene=dict(
            xaxis_title=x,
            yaxis_title=y,
            zaxis_title=z,
            # cube aspect
            aspectmode='cube' if equal_aspect else 'auto'
        ),
        showlegend=True)
        return fig