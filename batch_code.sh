echo '' > pos_res.txt
python3 lda_pca_sent.py ../dataset/economics/1377191648_sentiment_nuclear_power.csv tweet_text sentiment >> pos_res.txt
python3 lda_pca_sent.py ../dataset/economics/1377884570_tweet_global_warming.csv tweet existence >> pos_res.txt
python3 lda_pca_sent.py ../dataset/economics/1384367161_claritin_october_twitter_side_effects-1.csv content sentiment  >> pos_res.txt
python3 lda_pca_sent.py ../dataset/economics/Airline-Sentiment-2-w-AA.csv text airline_sentiment  >> pos_res.txt
python3 lda_pca_sent.py ../dataset/economics/Apple-Twitter-Sentiment-DFE.csv text sentiment >> pos_res.txt
python3 lda_pca_sent.py ../dataset/economics/Coachella-2015-2-DFE.csv text coachella_sentiment >> pos_res.txt
python3 lda_pca_sent.py ../dataset/economics/Deflategate-DFE.csv text deflate_sentiment >> pos_res.txt
python3 lda_pca_sent.py ../dataset/economics/GOP_REL_ONLY.csv text sentiment >> pos_res.txt
python3 lda_pca_sent.py ../dataset/economics/New-years-resolutions-DFE.csv text resolution_topics >> pos_res.txt
python3 lda_pca_sent.py ../dataset/economics/Political-media-DFE.csv message bias >> pos_res.txt
python3 lda_pca_sent.py ../dataset/economics/progressive-tweet-sentiment.csv tweet_text sentiment >> pos_res.txt
