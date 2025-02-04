def corr_plot(df):
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg')  # Set backend to non-interactive
    import matplotlib.pyplot as plt
    # df_clean = df.replace([np.inf, -np.inf], np.nan).dropna()
    print(df.describe())
    print(df.info())
    corr_matrix = df.corr(method='spearman')
    print(corr_matrix)
    g = sns.clustermap(
        corr_matrix,
        annot=False, 
        center=0,
        cmap='Blues',
        annot_kws={'size': 5},  # Smaller annotation text
        xticklabels=True,  # Force show all x labels
        yticklabels=True,  # Force show all y labels
        )
    g.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure to free memory