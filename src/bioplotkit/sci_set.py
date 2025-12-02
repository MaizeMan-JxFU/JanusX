def sci_set(font_path:str = None, font_family:str='Arial'):
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import font_manager
    # download the font files and save in this fold
    mpl.rcParams['pdf.fonttype'] = 42
    font_files = font_manager.findSystemFonts(fontpaths=font_path)
    plt.rc('font', family=font_family)
    for file in font_files:
        font_manager.fontManager.addfont(file) # 读取字体库

color_set = {
    0:['black','grey'],
    1:['red','green'],
    2:['green','orange'],
    3:['#C69287','#E4CD87'],
    4:['#E4391B','#F9992A'],
    5:["#3E4F94","#3E90BF"],
    6:['#714F91','#E4391B','#F9992A','#9C5E27','#739CCD','#398249'],
}
marker_set = ['o','+','x','8','s','p','P','D','2','1','^']