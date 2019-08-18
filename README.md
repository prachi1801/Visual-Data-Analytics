Demonstates three basic tasks of visual data analytics
 uses data of forest fires (fires.csv) begin with |N|≥500, |D|≥10)
 client-server system: python for processing (server), D3 for VIS (client)
Task 1: data clustering and decimation
 implemented random sampling and stratified sampling
 the latter includes the need for k-means clustering (optimize k using elbow)
Task 2: dimension reduction (use decimated data
 find the intrinsic dimensionality of the data using PCA
 produce scree plot visualization and mark the intrinsic dimensionality
 obtain the three attributes with highest PCA loadings
Task 3: visualization (use dimension reduced data)
 visualize the data projected into the top two PCA vectors via 2D scatterplot
 visualize the data via MDS (Euclidian & correlation distance) in 2D scatterplots
 visualize scatterplot matrix of the three highest PCA loaded attributes
