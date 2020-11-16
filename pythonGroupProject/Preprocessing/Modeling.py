scale = RobustScaler() 
df = scale.fit_transform(train)

pca = PCA().fit(df) # whiten=True
print('With only 120 features: {:6.4%}'.format(sum(pca.explained_variance_ratio_[:120])),"%\n")

print('After PCA, {:3} features only not explained {:6.4%} of variance ratio from the original {:3}'.format(120,(sum(pca.explained_variance_ratio_[120:])),df.shape[1]))
del df,all_data
																					

																					
