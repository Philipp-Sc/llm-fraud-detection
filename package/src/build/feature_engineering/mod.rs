use words_count::WordsCount;

lazy_static::lazy_static! {

        static ref RE_URL: regex::Regex = regex::Regex::new(r"(http|https|www)").unwrap();

        static ref RE_NUMBER: regex::Regex = regex::Regex::new(r"(\d)").unwrap();

        static ref RE_OBVIOUS_SPAM: regex::Regex = regex::Regex::new(r"(tokenomics|valid until|urgent|act now|limited time offer|free|win|cash|airdrop|rewards|claim|bonus|cheap|discount|debt|investment|lifetime|loans|marketing solution|mortgage rates|offer|opt in|pre-approved|quote|refinance|terms and conditions|trial|web traffic|work from home|buy direct|cancel at any time|check or money order|confidentiality|cures|direct email|direct marketing|internet marketing|lose weight|mass email|meet singles|no catch|no cost|no credit check|no fees|no gimmick|no hidden costs|no hidden fees|no interest|no investment|no obligation|no purchase necessary|no questions asked|no strings attached|not junk|notspam|obligation|passwords|requires initial investment|social security number|undisclosed|unsecured credit|unsecured debt|unsolicited|valium|viagra|vicodin|we hate spam|weight loss|xanax|#1|100% more|100% free|100% satisfied|additional income|be your own boss|best price|big bucks|billion|cash bonus|cents on the dollar|consolidate debt|double your cash|double your income|earn extra cash|earn money|eliminate bad credit|extra cash|extra income|expect to earn|fast cash|financial freedom|free access|free consultation|free gift|free hosting|free info|free investment|free membership|free money|free preview|free quote|free trial|full refund|get out of debt|get paid|giveaway|guaranteed|increase sales|increase traffic|incredible deal|lower rates|lowest price|make money|million dollars|miracle|money back|once in a lifetime|one time|pennies a day|potential earnings|prize|promise|pure profit|risk-free|satisfaction guaranteed|save big money|save up to|special promotion)").unwrap();

        static ref RE_UPPER_CASE_WORD: regex::Regex = regex::Regex::new(r"\b[A-Z]+\b").unwrap();

        static ref RE_OBVIOUS_NOT_SPAM: regex::Regex = regex::Regex::new(r"(acquire|admire|appreciate|authentic|beauty|celebrate|cleanse|comfortable|compassion|contribute|create|delicious|delightful|discover|education|elegant|enjoy|explore|freedom|genuine|graceful|growth|happiness|healthy|honesty|hope|improve|inspire|integrity|joyful|knowledge|learn|love|mindful|natural|nourish|opportunity|passion|peaceful|progress|respect|satisfy|serene|sincere|smile|support|thankful|thrive|trustworthy|unique|valuable|vibrant|wisdom|wonderful|worthwhile|bright|care|cheerful|clever|connect|cozy|cute|effort|energy|fair|friendly|funny|generous|gentle|gift|glad|good|grateful|happy|help|idea|insight|involve|jolly|kind|laugh|listen|lovely|lucky|nice|optimism|organize|partner|playful|polite|pretty|radiant|share|smart|smooth|success|together|understand|victory|warm|welcome)").unwrap();
     }

// TODO: bag of words, most common words. traditional NLP, 
pub fn get_features(text: String) -> Vec<f64> {

    let mut features = Vec::new();
    
    let word_count = words_count::count(&text);
    // Number of Words
    features.push(word_count.words as f64);
    // Number of URLs
    features.push(RE_URL.captures_iter(&text.to_lowercase()).count() as f64);
    // Number of Spam Words
    features.push(RE_OBVIOUS_SPAM.captures_iter(&text.to_lowercase()).count() as f64);
    // Number of Non-Spam Words
    features.push(RE_OBVIOUS_NOT_SPAM.captures_iter(&text.to_lowercase()).count() as f64);
    // Number of Upper-Case Words
    features.push(RE_UPPER_CASE_WORD.captures_iter(&text).count() as f64);

    // replace "Number of Spam Words" and "Number of Non-Spam Words" with Naive Bayes Prediction of CountVectors.

    features
}
