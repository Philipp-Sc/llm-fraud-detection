use words_count::WordsCount;

lazy_static::lazy_static! {

        static ref RE_URL: regex::Regex = regex::Regex::new(r"(https?|ftp)").unwrap();

        static ref RE_NUMBER: regex::Regex = regex::Regex::new(r"(\d)").unwrap();

    }

// makes no meaningful difference
// TODO: bag of words, most common words. traditional NLP, 
pub fn get_features(text: String) -> Vec<f64> {

    let mut features = Vec::new();
    
    //let word_count = words_count::count(&text);
    //features.push(word_count.characters as f64/word_count.words as f64); // avg word len

    //features.push(text.chars().count() as f64); // count chars
    //features.push(text.split_whitespace().count() as f64); // whitespace
    
    //features.push(RE_NUMBER.captures_iter(&text).count() as f64); // count numbers
    //features.push(RE_URL.captures_iter(&text).count() as f64); // count urls

    features
}
