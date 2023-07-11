use importance::score::Model;
use words_count::WordsCount;
use crate::build::classification::deep_learning::MockModel;

lazy_static::lazy_static! {

        static ref RE_URL: regex::Regex = regex::Regex::new(r"(http|https|www)").unwrap();

        static ref RE_UPPER_CASE_WORD: regex::Regex = regex::Regex::new(r"\b[A-Z]+\b").unwrap();
        
        static ref RE_NON_STANDARD: regex::Regex = regex::Regex::new(r"[^\w\s]").unwrap();
        
        static ref RE_PUNCTUATION: regex::Regex = regex::Regex::new(r"[[:punct:]]+").unwrap();

        static ref RE_EMOJI: regex::Regex = regex::Regex::new(r"\p{Emoji}").unwrap();
     }

pub fn get_hard_coded_feature_labels() -> Vec<String> {
    vec![
        "categorical_nb_model_predict",
        "gaussian_nb_model_predict",
        "word_count.words",
        "word_count.characters",
        "word_count.whitespaces",
        "word_count.cjk",
        "RE_URL",
        "RE_UPPER_CASE_WORD",
        "RE_NON_STANDARD",
        "RE_PUNCTUATION",
        "RE_EMOJI"
    ].into_iter().map(|x| x.to_string()).collect()
}

pub fn get_features(text: &String, topic_predictions: Vec<f64>, sentiment_prediction: f64, custom_features: bool, topics: bool, latent_variables: bool) -> Vec<f64> {

    let mut features = Vec::new();

    if topics {
        features.append(&mut topic_predictions.clone());
    }
    if latent_variables {
        // Sentiment
        features.push(sentiment_prediction);
        // Naive Bayes
        features.push(super::naive_bayes::categorical_nb_model_predict(vec![text.clone()]).unwrap()[0] as f64);
        features.push(super::naive_bayes::gaussian_nb_model_predict(vec![text.clone()]).unwrap()[0] as f64);
        // NN
        //let model = MockModel{ label: "./NeuralNetSelectedTopics.bin".to_string()};
        //features.append(&mut model.predict(&vec![topic_predictions.clone()]));
        // NN
        //let model = MockModel{ label: "./NeuralNetCustomFeatures.bin".to_string()};
        //features.append(&mut model.predict(&vec![topic_predictions.clone()]));
        // RandomForest
        // ...

    }

    if custom_features {
        let word_count = words_count::count(text);
        features.push(word_count.words as f64); // 8
        features.push(word_count.characters as f64); // 3
        features.push(word_count.whitespaces as f64); // 5
        features.push(word_count.cjk as f64);

        features.push(RE_URL.captures_iter(&text.to_lowercase()).count() as f64); // 9
        features.push(RE_UPPER_CASE_WORD.captures_iter(text).count() as f64);
        features.push(RE_NON_STANDARD.captures_iter(text).count() as f64); // 7
        features.push(RE_PUNCTUATION.captures_iter(text).count() as f64); // 4
        features.push(RE_EMOJI.captures_iter(text).count() as f64); // 6
    }

    features
}
