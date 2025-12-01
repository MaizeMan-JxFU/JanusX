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