# TwitterCoordination
This repo contains the files I used in Chapter 2 of my Master's Thesis. 


## Main Files 
The main python files I used are below. Some are stand alone scripts, others are imported in the notebooks in the next section.

- __preprocessing.py__: Functions for loading dataset
- __trending.py__: Functions for building time series dataframes and integrating trending data
- __exposure_script.py__: Script for calculating number of follower network exposures before first tweet
- __cascade.py__: Class for creating retweet cascades 
- __build_cascades.py__: Script for meta-analysis of different retweet attribution methods 
- __nearest_neighbors_revised.py__: Script for encoding sentences using SBERT and finding nearest neighbors
- __config.py__: Defines some grid directory locations
- __imports.py__: Common imports

### Main Notebooks
The notebooks with my final results begin with THESIS.
- __THESIS_Preliminary_Findings.ipynb__: Final visualizations for first parts of thesis (retweet cascades, exposure curves, nlp stuff)
- __THESIS_Trending_Effect_Table_Regressions.ipynb__: Notebook with regression analysis
- __THESIS_Specification_Curve_2.ipynb__: Spec curve analysis
- __THESIS_Trending_Effect_Qualitative.ipynb__: Work on undwerstanding network characteristics of people possible exposed through Trending Topics page.

### Notebook Directories
There are many more notebooks exploring different avenues. I moved them into separate directories. 
1. __Preliminary__: This contains notebooks for analysis of cascades, retweets, and exposure curves.
2. __NLP__: This contains notebooks for analyzing sentence embeddings of tweets. 
3. __Trending__: This contains notebooks for quantifying the return to the Trending Topics page. 
4. __Other__: misc files. 
