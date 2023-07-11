use std::collections::HashSet;
use importance::score::Model;
use words_count::WordsCount;
use crate::build::classification::deep_learning::NNMockModel;
use linkify::LinkFinder;
use crate::build::classification::ClassificationMockModel;


lazy_static::lazy_static! {

        static ref RE_UPPER_CASE_WORD: regex::Regex = regex::Regex::new(r"\b[A-Z]+\b").unwrap();
        
        static ref RE_NON_STANDARD: regex::Regex = regex::Regex::new(r"[^\w\s]").unwrap();
        
        static ref RE_PUNCTUATION: regex::Regex = regex::Regex::new(r"[[:punct:]]+").unwrap();

        static ref RE_EMOJI: regex::Regex = regex::Regex::new(r"\p{Emoji}").unwrap();

        static ref LINK_FINDER: LinkFinder = get_link_finder();

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
        // Topics
        features.append(&mut topic_predictions.clone());
        // Sentiment
        features.push(sentiment_prediction);
    }
    let mut custom_feature_vec = get_custom_features(text);
    if custom_features {
        features.append(&mut custom_feature_vec);
    }

    if latent_variables {
        // Naive Bayes
        features.push(super::naive_bayes::categorical_nb_model_predict(vec![text.clone()]).unwrap()[0] as f64);
        features.push(super::naive_bayes::gaussian_nb_model_predict(vec![text.clone()]).unwrap()[0] as f64);
        // NN
        let model = NNMockModel { label: "./NeuralNetTopicsAndCustomFeatures.bin".to_string()};

        let mut input: Vec<Vec<f64>> = Vec::new();
        input.push(topic_predictions.clone());
        input[0].push(sentiment_prediction);
        input[0].append(&mut custom_feature_vec.clone());

        features.append(&mut model.predict(&input));
        // RandomForest
        let model = ClassificationMockModel { label: "./RandomForestRegressorTopicsAndCustomFeatures.bin".to_string()};
        features.append(&mut model.predict(&input));


    }

    features
}

fn get_custom_features(text: &String) -> Vec<f64> {
    let mut features = Vec::new();
    let word_count = words_count::count(text);
    features.push(word_count.words as f64); // 8
    features.push(word_count.characters as f64); // 3
    features.push(word_count.whitespaces as f64); // 5
    features.push(word_count.cjk as f64);

    features.push(extract_links(text).into_iter().count() as f64); // 9
    features.push(RE_UPPER_CASE_WORD.captures_iter(text).count() as f64);
    features.push(RE_NON_STANDARD.captures_iter(text).count() as f64); // 7
    features.push(RE_PUNCTUATION.captures_iter(text).count() as f64); // 4
    features.push(RE_EMOJI.captures_iter(text).count() as f64); // 6

    features
}


fn extract_links(text: &String) -> Vec<String> {

    let links = LINK_FINDER.links(&text);
    let mut output: Vec<String> = Vec::new();
    for link in links  {
        if let s = link.as_str().to_string() {
            if s.parse::<f64>().is_err() && s.chars().count() >= 8 && s.chars().any(|c| c.is_alphabetic()) {
                output.push(s);
            }
        }
    }
    // Convert the vector to a HashSet
    let set: HashSet<String> = output.into_iter().collect();
    // Convert the HashSet back to a vector
    set.into_iter().collect()
}

fn get_link_finder() -> LinkFinder {
    let mut finder = LinkFinder::new();
    finder.url_must_have_scheme(false);
    finder
}