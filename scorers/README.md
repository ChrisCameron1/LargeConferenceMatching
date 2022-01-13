# LargeConferenceMatching
Each paper-reviewer score is computed by combining the following 4 scores :

1. ACL Paper Matching Score
2. Keyword Matching Score
3. TPMS Score
4. Bid Score

- ACL score, keyword score and TPMS score are aggregated into a single number between 0 and 1. 
- The overall score is then computed by raising the aggregated score to a power encoding the reviewer's bid: *20*, *1*, *2/3*, *0.4*, *0.25* corresponding to *not-willing*, *not-entered*, *in-a-pinch*, *willing*, and *eager* bids respectively. 

# Steps to Compute the Overall Score
1. Compute the ACL scores (refer to the `acl-scorer` folder)
2. Compute the keyword scores (refer to the `keyword-scorer` folder)
3. Retrieve the TPMS scores through CMT
4. Compute the aggregated score using `score_normalization.ipynb`
5. Compute the overall score using the bid information and the aggregate scores.