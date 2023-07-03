use words_count::WordsCount;

lazy_static::lazy_static! {

        static ref RE_URL: regex::Regex = regex::Regex::new(r"(http|https|www)").unwrap();

        static ref RE_NUMBER: regex::Regex = regex::Regex::new(r"\d+").unwrap();

        static ref RE_UPPER_CASE_WORD: regex::Regex = regex::Regex::new(r"\b[A-Z]+\b").unwrap();
        
        static ref RE_NON_STANDARD: regex::Regex = regex::Regex::new(r"[^\w\s]").unwrap();
        
        static ref RE_PUNCTUATION: regex::Regex = regex::Regex::new(r"[[:punct:]]+").unwrap();

        static ref RE_EMOJI: regex::Regex = regex::Regex::new(r"\p{Emoji}").unwrap();
     }

pub fn get_features(text: String) -> Vec<f64> {

    let mut features = Vec::new();
    
    // Naive Bayes Spam Prediction
    features.push(super::naive_bayes::categorical_nb_model_predict(vec![text.clone()]).unwrap()[0] as f64);
    features.push(super::naive_bayes::gaussian_nb_model_predict(vec![text.clone()]).unwrap()[0] as f64);


    let word_count = words_count::count(&text);
    features.push(word_count.words as f64);
    features.push(word_count.characters as f64);
    features.push(word_count.whitespaces as f64);
    features.push(word_count.cjk as f64);

    features.push(RE_URL.captures_iter(&text.to_lowercase()).count() as f64);
    features.push(RE_UPPER_CASE_WORD.captures_iter(&text).count() as f64);
    features.push(RE_NON_STANDARD.captures_iter(&text).count() as f64);
    features.push(RE_PUNCTUATION.captures_iter(&text).count() as f64);
    features.push(RE_EMOJI.captures_iter(&text).count() as f64);

    features
}
