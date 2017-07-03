--(Load Yelp restaurants review) 
Restaurant = load '../yelp_1000_dataset/part-r-00000' USING JsonLoader();

--(Yelp review text with review ID)
Rest_with_reviewid = foreach Restaurant generate RANDOM() as rnd, restaurant_review::bus_id, restaurant_review::usr_id, restaurant_review::rating, restaurant_review::dat, restaurant_review::review_text;
Review_words = foreach Rest_with_reviewid generate rnd, restaurant_review::bus_id, restaurant_review::usr_id, restaurant_review::rating, restaurant_review::dat,FLATTEN(TOKENIZE(REPLACE(LOWER(TRIM(restaurant_review::review_text)),'[\\p{Punct},\\p{Cntrl}]',''))) AS review_token;

--(Business ID, UserID, rating, date, review text)
review_token_trim = foreach Review_words generate rnd,restaurant_review::bus_id, restaurant_review::usr_id, restaurant_review::rating, restaurant_review::dat, TRIM(review_token) as rword;

--(Review ID, Review token)
review_tokenize = foreach review_token_trim generate rnd, rword;

--(Remove stop words in Review tokens)
stop_words = LOAD 'stopwords.txt' AS (sword:chararray);
sword_join = JOIN review_tokenize BY rword LEFT OUTER, stop_words BY sword; 
sword_filter = FILTER sword_join BY stop_words::sword IS NULL;

--(Extract Postive words from Review text)
pdictionary = load 'positive-words.txt'AS (pword:chararray);
pjoin = join sword_filter by rword, pdictionary by pword ;
pword_grp = group pjoin by rnd;
pword_count = foreach pword_grp generate group,COUNT($1.$1);

--(Extract Negative words from Review text)
ndictionary = load 'negative-words.txt'AS (nword:chararray);
njoin = join sword_filter by rword , ndictionary by nword ;
nword_grp = group njoin by rnd;
nword_count = foreach nword_grp generate group,COUNT($1.$1) ;

--(Join the results)
neg_pos_words = JOIN pword_count by group, nword_count by group;
NP_words = foreach neg_pos_words generate $0 as rev_id,$1,$3;
Complete_data = join NP_words by rev_id, Rest_with_reviewid by rnd;
final_result = foreach Complete_data generate $0,$6,$8,ROUND(2.5+($1-$2)*2.5/($1+$2));
Store final_result INTO 'Yelpdata_Stars';

